from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.datamodule import ImageAuthorshipDataModule, TripadvisorImageAuthorshipBPRDataset
from src.models.presley import PRESLEY
from src.callbacks import EmissionsTrackerCallback
from src.centroids import get_centroid_preds
from src.test import test_tripadvisor_authorship_task


# =========================
# φ / ρ building blocks
# =========================
def phi_mlp_2ln(d: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(1536, d),
        nn.ReLU(inplace=True),
        nn.Linear(d, d),
        nn.LayerNorm(d),
    )


def phi_linear(d: int) -> nn.Module:
    return nn.Linear(1536, d)


def phi_identity() -> nn.Module:
    return nn.Identity()


def rho_mlp_2ln_dd(d: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d, d),
        nn.ReLU(inplace=True),
        nn.Linear(d, d),
        nn.LayerNorm(d),
    )


def rho_mlp_2ln_1536d(d: int) -> nn.Module:
    # Para φ=Identity => pooled queda en 1536, ρ debe arrancar en 1536→d
    return nn.Sequential(
        nn.Linear(1536, d),
        nn.ReLU(inplace=True),
        nn.Linear(d, d),
        nn.LayerNorm(d),
    )


def rho_linear_dd(d: int) -> nn.Module:
    return nn.Linear(d, d)


def rho_linear_1536d(d: int) -> nn.Module:
    return nn.Linear(1536, d)


def build_variant(variant: str, d: int) -> Tuple[nn.Module, Optional[nn.Module], bool]:
    """
    Returns (phi, rho, ds_no_rho)
    ds_no_rho=True => DeepSetEmbeddingBlock hace skip total de rho.
    """
    v = variant.lower()

    if v in {"baseline", "base"}:
        return phi_mlp_2ln(d), rho_mlp_2ln_dd(d), False

    if v in {"phi_simple", "v1"}:
        return phi_linear(d), rho_mlp_2ln_dd(d), False

    if v in {"rho_simple", "v2"}:
        return phi_mlp_2ln(d), rho_linear_dd(d), False

    if v in {"both_simple", "v3"}:
        return phi_linear(d), rho_linear_dd(d), False

    if v in {"no_rho", "v4"}:
        return phi_mlp_2ln(d), None, True

    if v in {"no_phi", "v5"}:
        return phi_identity(), rho_mlp_2ln_1536d(d), False

    if v in {"no_rho_phi_simple", "v6"}:
        return phi_linear(d), None, True

    if v in {"no_phi_rho_simple", "v7"}:
        return phi_identity(), rho_linear_1536d(d), False

    raise ValueError(f"Unknown variant: {variant}")


def sanity_check_shapes(phi: nn.Module, rho: Optional[nn.Module], use_rho: bool, d: int, k: int) -> None:
    # Check rápido para evitar mismatches evidentes
    with torch.no_grad():
        user_images = torch.zeros(2, k, 1536)
        user_masks = torch.ones(2, k)

        phi_x = phi(user_images)
        if phi_x.dim() != 3:
            raise RuntimeError(f"phi output must be (B,K,dim). Got {tuple(phi_x.shape)}")

        pooled = (phi_x * user_masks.unsqueeze(-1)).mean(dim=1)
        out = pooled if (not use_rho) else rho(pooled)  # type: ignore[arg-type]

        if out.shape[-1] != d:
            raise RuntimeError(f"Final user embedding dim != d. Got {tuple(out.shape)}, expected last_dim={d}.")


def _copy_ckpt(src_path: str, dst_path: Path) -> Path:
    if not src_path:
        raise FileNotFoundError("Lightning did not provide a checkpoint path (empty).")
    sp = Path(src_path)
    if not sp.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {sp}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sp, dst_path)
    return dst_path


def extract_metric(metrics_dict: Dict[str, Any], model_name: str) -> Dict[str, Optional[float]]:
    m = metrics_dict.get(model_name, {}) if isinstance(metrics_dict, dict) else {}

    def get_any(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in m and m[k] is not None:
                try:
                    return float(m[k])
                except Exception:
                    return None
        return None

    return {
        "MRecall@10": get_any(["recall@10_10plus", "recall@10"]),
        "MNDCG@10": get_any(["ndcg@10_10plus", "ndcg@10"]),
        "MAUC": get_any(["auc_all", "auc", "auc_10plus"]),
    }


@dataclass
class RunCfg:
    city: str = "barcelona"
    model: str = "PRESLEY"
    batch_size: int = 8192
    max_user_images: int = 20
    workers: int = 0

    # Ablación: TRAIN/DEV conjunto + sin validación interna
    use_train_val: bool = True
    no_validation: bool = True

    max_epochs: int = 75

    # Solo si activases validación
    patience: int = 10
    min_delta: float = 1e-4

    seed: int = 0
    limit_users: Optional[int] = None
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    limit_test_batches: Optional[int] = None


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()

    # Best HPs (default)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--city", type=str, default="barcelona")
    parser.add_argument("--model_name", type=str, default="PRESLEY")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_user_images", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=75)

    # Defaults como quieres: TRAIN/DEV + no_validation
    parser.add_argument("--use_train_val", dest="use_train_val", action="store_true")
    parser.add_argument("--no_use_train_val", dest="use_train_val", action="store_false")
    parser.set_defaults(use_train_val=True)

    parser.add_argument("--no_validation", dest="no_validation", action="store_true")
    parser.add_argument("--with_validation", dest="no_validation", action="store_false")
    parser.set_defaults(no_validation=True)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    parser.add_argument("--limit_users", type=int, default=None)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--limit_test_batches", type=int, default=None)

    # Por defecto EXCLUYE baseline (porque ya lo tienes)
    parser.add_argument(
        "--variants",
        type=str,
        default="phi_simple,rho_simple,both_simple,no_rho,no_phi,no_rho_phi_simple,no_phi_rho_simple",
    )

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_true", default=False)

    args = parser.parse_args()
    resume = args.resume and (not args.no_resume)

    cfg = RunCfg(
        city=args.city,
        model=args.model_name,
        batch_size=args.batch_size,
        max_user_images=args.max_user_images,
        workers=args.workers,
        use_train_val=args.use_train_val,
        no_validation=args.no_validation,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        limit_users=args.limit_users,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise ValueError("No variants provided.")

    # >>> Carpeta como pides:
    # results/ablation/<city>/<model>/...
    ablation_root = Path("results") / "ablation" / cfg.city / cfg.model
    ablation_root.mkdir(parents=True, exist_ok=True)


    out_csv = ablation_root / "ablation_summary.csv"
    out_json = ablation_root / "ablation_summary.json"

    summary_rows: List[Dict[str, Any]] = []

    for variant in variants:
        # Misma seed para TODAS las variantes => comparabilidad
        pl.seed_everything(cfg.seed, workers=True)

        phi, rho, ds_no_rho = build_variant(variant, args.d)
        use_rho = not ds_no_rho
        sanity_check_shapes(phi=phi, rho=rho, use_rho=use_rho, d=args.d, k=cfg.max_user_images)

        run_id = (
            f"ABL__{variant}"
            f"__d{args.d}__lr{args.lr}__do{args.dropout}__bs{cfg.batch_size}"
            f"__K{cfg.max_user_images}"
            f"__tv{int(cfg.use_train_val)}__noval{int(cfg.no_validation)}"
            f"__rho{int(use_rho)}__seed{cfg.seed}"
        )
        run_dir = ablation_root / run_id
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = run_dir / "test_metrics.json"
        failed_flag = run_dir / "_FAILED.txt"

        if resume and metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            row = {
                "variant": variant,
                "run_id": run_id,
                "seed": cfg.seed,
                "d": args.d,
                "lr": args.lr,
                "dropout": args.dropout,
                "phi_cfg": variant,  # placeholder (luego en memoria pones columnas)
                "rho_cfg": variant,
                "use_rho": int(use_rho),
                "status": "skipped(existing)",
            }
            row.update(extract_metric(metrics, cfg.model))
            summary_rows.append(row)

            try:
                import pandas as pd
                pd.DataFrame(summary_rows).to_csv(out_csv, index=False, encoding="utf-8")
            except Exception:
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(summary_rows, f, indent=2)
            continue

        if failed_flag.exists():
            failed_flag.unlink(missing_ok=True)

        cfg_dump = {
            "variant": variant,
            "city": cfg.city,
            "model": cfg.model,
            "seed": cfg.seed,
            "d": args.d,
            "lr": args.lr,
            "dropout": args.dropout,
            "batch_size": cfg.batch_size,
            "max_user_images": cfg.max_user_images,
            "workers": cfg.workers,
            "use_train_val": cfg.use_train_val,
            "no_validation": cfg.no_validation,
            "ds_no_rho": ds_no_rho,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "min_delta": cfg.min_delta,
            "limit_users": cfg.limit_users,
            "limit_train_batches": cfg.limit_train_batches,
            "limit_val_batches": cfg.limit_val_batches,
            "limit_test_batches": cfg.limit_test_batches,
        }
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_dump, f, indent=2)

        t0 = time.time()

        try:
            dm = ImageAuthorshipDataModule(
                city=cfg.city,
                batch_size=cfg.batch_size,
                num_workers=cfg.workers,
                dataset_class=TripadvisorImageAuthorshipBPRDataset,
                use_train_val=cfg.use_train_val,
                limit_users=cfg.limit_users,
                model_name=cfg.model,
                max_user_images=cfg.max_user_images,
                no_validation=cfg.no_validation,
            )
            dm.setup()

            logger = TensorBoardLogger(save_dir=str(run_dir / "tb"), name="", version="")

            callbacks: List[Any] = []

            if cfg.no_validation:
                # >>> LO QUE TE IMPORTA:
                # Guardar SIEMPRE "last"
                ckpt_cb = ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    save_last=True,
                    save_top_k=0,
                    every_n_epochs=1,
                    save_on_train_epoch_end=True,
                )
                callbacks.append(ckpt_cb)
            else:
                ckpt_cb = ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    filename="best",
                    save_top_k=1,
                    monitor="val_auc",
                    mode="max",
                    save_on_train_epoch_end=False,
                )
                early = EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    min_delta=cfg.min_delta,
                    patience=cfg.patience,
                    check_on_train_epoch_end=False,
                )
                callbacks.extend([ckpt_cb, early])

            callbacks.append(EmissionsTrackerCallback(log_to_trainer=True))
            callbacks.append(LearningRateMonitor(logging_interval="step"))

            trainer_kwargs: Dict[str, Any] = dict(
                max_epochs=cfg.max_epochs,
                accelerator="auto",
                strategy="auto",
                devices="auto",
                callbacks=callbacks,
                logger=logger,
                enable_progress_bar=True,
                log_every_n_steps=1,
                enable_model_summary=True,
                num_sanity_val_steps=0 if cfg.no_validation else 2,
            )
            if cfg.limit_train_batches is not None:
                trainer_kwargs["limit_train_batches"] = cfg.limit_train_batches
            if cfg.limit_val_batches is not None:
                trainer_kwargs["limit_val_batches"] = cfg.limit_val_batches

            trainer = pl.Trainer(**trainer_kwargs)

            model = PRESLEY(
                d=args.d,
                nusers=1,
                lr=args.lr,
                phi=phi,
                rho=rho,
                dropout=args.dropout,
                ds_no_rho=ds_no_rho,
            )

            trainer.fit(model=model, datamodule=dm)

            # =========================
            # CHECKPOINT (no_validation)
            # =========================
            # Lightning puede dejarlo como last.ckpt o last-v1.ckpt. No nos fiamos del nombre.
            # Cogemos el path REAL y lo copiamos a checkpoints/last.ckpt (estable).
            last_real = getattr(ckpt_cb, "last_model_path", "")
            stable_last = ckpt_dir / "last.ckpt"
            ckpt_path = _copy_ckpt(last_real, stable_last)

            # =========================
            # EVAL (cargar last.ckpt)
            # =========================
            phi2, rho2, ds_no_rho2 = build_variant(variant, args.d)
            model_eval = PRESLEY(
                d=args.d,
                nusers=1,
                lr=args.lr,
                phi=phi2,
                rho=rho2,
                dropout=args.dropout,
                ds_no_rho=ds_no_rho2,
            )

            ckpt_obj = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj

            incomp = model_eval.load_state_dict(state_dict, strict=False)
            if incomp.missing_keys or incomp.unexpected_keys:
                raise RuntimeError(
                    f"Checkpoint incompatible.\nMissing: {incomp.missing_keys}\nUnexpected: {incomp.unexpected_keys}"
                )

            model_eval.eval()

            pred_trainer = pl.Trainer(
                accelerator="auto",
                devices="auto",
                logger=False,
                enable_progress_bar=False,
            )

            preds_list = pred_trainer.predict(model=model_eval, datamodule=dm)
            if not isinstance(preds_list, list) or len(preds_list) == 0:
                raise RuntimeError("predict() returned empty outputs.")
            test_preds = torch.cat([p.detach().cpu() for p in preds_list], dim=0).numpy()

            # Baselines reproducibles
            torch.manual_seed(cfg.seed)
            rnd = torch.mean(torch.rand((len(dm.test_dataset), 10)), dim=1).cpu().numpy()

            cnt = get_centroid_preds(dm)
            cnt = cnt.cpu().numpy() if torch.is_tensor(cnt) else cnt

            models_preds = {
                cfg.model: test_preds,
                "RANDOM": rnd,
                "CNT": cnt,
            }

            class SafeArgs:
                def __init__(self, **kw):
                    self.__dict__.update(kw)

                def __getattr__(self, item):
                    return None

            test_args = SafeArgs(
                city=cfg.city,
                model=[cfg.model],
                seed=cfg.seed,
                no_validation=cfg.no_validation,
                use_train_val=cfg.use_train_val,
                max_user_images=cfg.max_user_images,
            )

            metrics = test_tripadvisor_authorship_task(dm, models_preds, test_args)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            dt = time.time() - t0

            row = {
                "variant": variant,
                "run_id": run_id,
                "seed": cfg.seed,
                "d": args.d,
                "lr": args.lr,
                "dropout": args.dropout,
                "use_rho": int(use_rho),
                "ckpt_used": str(ckpt_path),
                "wall_time_s": round(dt, 2),
                "status": "ok",
            }
            row.update(extract_metric(metrics, cfg.model))
            summary_rows.append(row)

        except Exception as e:
            with open(failed_flag, "w", encoding="utf-8") as f:
                f.write(str(e))
            summary_rows.append(
                {
                    "variant": variant,
                    "run_id": run_id,
                    "seed": cfg.seed,
                    "d": args.d,
                    "lr": args.lr,
                    "dropout": args.dropout,
                    "use_rho": int(use_rho),
                    "status": f"FAILED: {type(e).__name__}",
                    "error": str(e),
                }
            )

        finally:
            for obj_name in ["model", "model_eval", "trainer", "dm"]:
                if obj_name in locals():
                    try:
                        del locals()[obj_name]
                    except Exception:
                        pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Dump incremental
        try:
            import pandas as pd
            pd.DataFrame(summary_rows).to_csv(out_csv, index=False, encoding="utf-8")
        except Exception:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary_rows, f, indent=2)

    print(f"[DONE] Summary saved in: {out_csv}")


if __name__ == "__main__":
    main()
