from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
from torch import nn

from src.datamodule import ImageAuthorshipDataModule, TripadvisorImageAuthorshipBPRDataset
from src.models.presley import PRESLEY
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


class SafeArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, item):
        return None


@dataclass
class EvalCfg:
    city: str = "barcelona"
    model: str = "PRESLEY"

    # Deben coincidir con el entrenamiento
    d: int = 64
    lr: float = 1e-3
    dropout: float = 0.4
    seed: int = 0

    batch_size: int = 8192
    max_user_images: int = 20
    workers: int = 0

    use_train_val: bool = True
    no_validation: bool = True

    # Rutas
    ablation_root: Path = Path("results") / "ablation"


def _read_variant_from_run_id(run_id: str) -> str:
    # run_id esperado: ABL__{variant}__d64__lr...
    parts = run_id.split("__")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Cannot parse variant from run_id: {run_id}")
    return parts[1]


def _find_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints dir: {ckpt_dir}")

    # Preferimos last.ckpt si existe
    stable = ckpt_dir / "last.ckpt"
    if stable.exists():
        return stable

    # Si no, busca cualquier last*.ckpt (last-v1.ckpt, etc.)
    cands = sorted(ckpt_dir.glob("last*.ckpt"))
    if cands:
        return cands[-1]

    # Último recurso: cualquier .ckpt
    cands = sorted(ckpt_dir.glob("*.ckpt"))
    if cands:
        return cands[-1]

    raise FileNotFoundError(f"No checkpoint found in: {ckpt_dir}")


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Evaluate existing ablation checkpoints (no training).")
    parser.add_argument("--city", type=str, default="barcelona")
    parser.add_argument("--model_name", type=str, default="PRESLEY")

    # Si NO quieres flags, no los uses. Están para casos raros.
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_user_images", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--use_train_val", dest="use_train_val", action="store_true")
    parser.add_argument("--no_use_train_val", dest="use_train_val", action="store_false")
    parser.set_defaults(use_train_val=True)

    parser.add_argument("--no_validation", dest="no_validation", action="store_true")
    parser.add_argument("--with_validation", dest="no_validation", action="store_false")
    parser.set_defaults(no_validation=True)

    args = parser.parse_args()

    cfg = EvalCfg(
        city=args.city,
        model=args.model_name,
        d=args.d,
        lr=args.lr,
        dropout=args.dropout,
        seed=args.seed,
        batch_size=args.batch_size,
        max_user_images=args.max_user_images,
        workers=args.workers,
        use_train_val=args.use_train_val,
        no_validation=args.no_validation,
        ablation_root=Path("results") / "ablation",
    )

    # results/ablation/<city>/<model>/
    root = cfg.ablation_root / cfg.city / cfg.model
    if not root.exists():
        raise FileNotFoundError(f"Ablation root not found: {root}")

    # Encuentra runs ABL__*
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("ABL__")])
    if not run_dirs:
        raise RuntimeError(f"No ABL__ run dirs found under: {root}")

    out_csv = root / "ablation_eval_summary.csv"
    out_json = root / "ablation_eval_summary.json"

    # Carga datamodule UNA VEZ (mismo test set para todos)
    pl.seed_everything(cfg.seed, workers=True)
    dm = ImageAuthorshipDataModule(
        city=cfg.city,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        dataset_class=TripadvisorImageAuthorshipBPRDataset,
        use_train_val=cfg.use_train_val,
        limit_users=None,
        model_name=cfg.model,
        max_user_images=cfg.max_user_images,
        no_validation=cfg.no_validation,
    )
    dm.setup()

    # Baselines (una vez)
    torch.manual_seed(cfg.seed)
    rnd = torch.mean(torch.rand((len(dm.test_dataset), 10)), dim=1).cpu().numpy()

    cnt = get_centroid_preds(dm)
    cnt = cnt.cpu().numpy() if torch.is_tensor(cnt) else cnt

    test_args = SafeArgs(
        city=cfg.city,
        model=[cfg.model],
        seed=cfg.seed,
        no_validation=cfg.no_validation,
        use_train_val=cfg.use_train_val,
        max_user_images=cfg.max_user_images,
    )

    pred_trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_progress_bar=False,
    )

    summary_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        metrics_path = run_dir / "test_metrics.json"

        # Si ya existe metrics, lo reutilizamos (evita repetir eval)
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                row = {
                    "variant": _read_variant_from_run_id(run_id),
                    "run_id": run_id,
                    "seed": cfg.seed,
                    "d": cfg.d,
                    "lr": cfg.lr,
                    "dropout": cfg.dropout,
                    "status": "skipped(existing_metrics)",
                }
                row.update(extract_metric(metrics, cfg.model))
                summary_rows.append(row)
                continue
            except Exception:
                # si está corrupto, reevaluamos
                pass

        try:
            ckpt_path = _find_ckpt(run_dir)
            variant = _read_variant_from_run_id(run_id)

            phi, rho, ds_no_rho = build_variant(variant, cfg.d)

            model_eval = PRESLEY(
                d=cfg.d,
                nusers=1,
                lr=cfg.lr,
                phi=phi,
                rho=rho,
                dropout=cfg.dropout,
                ds_no_rho=ds_no_rho,
            )

            ckpt_obj = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj

            incomp = model_eval.load_state_dict(state_dict, strict=False)
            if incomp.missing_keys or incomp.unexpected_keys:
                raise RuntimeError(
                    f"Checkpoint incompatible.\nMissing: {incomp.missing_keys}\nUnexpected: {incomp.unexpected_keys}"
                )

            model_eval.eval()

            preds_list = pred_trainer.predict(model=model_eval, datamodule=dm)
            if not isinstance(preds_list, list) or len(preds_list) == 0:
                raise RuntimeError("predict() returned empty outputs.")
            test_preds = torch.cat([p.detach().cpu() for p in preds_list], dim=0).numpy()

            models_preds = {
                cfg.model: test_preds,
                "RANDOM": rnd,
                "CNT": cnt,
            }

            metrics = test_tripadvisor_authorship_task(dm, models_preds, test_args)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            row = {
                "variant": variant,
                "run_id": run_id,
                "seed": cfg.seed,
                "d": cfg.d,
                "lr": cfg.lr,
                "dropout": cfg.dropout,
                "ckpt_used": str(ckpt_path),
                "status": "ok",
            }
            row.update(extract_metric(metrics, cfg.model))
            summary_rows.append(row)

        except Exception as e:
            summary_rows.append(
                {
                    "variant": run_dir.name,
                    "run_id": run_id,
                    "seed": cfg.seed,
                    "d": cfg.d,
                    "lr": cfg.lr,
                    "dropout": cfg.dropout,
                    "status": f"FAILED: {type(e).__name__}",
                    "error": str(e),
                }
            )

        # dump incremental (para no perderlo si peta a mitad)
        try:
            import pandas as pd

            pd.DataFrame(summary_rows).to_csv(out_csv, index=False, encoding="utf-8")
        except Exception:
            out_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"[DONE] Eval summary saved in: {out_csv}")


if __name__ == "__main__":
    main()
