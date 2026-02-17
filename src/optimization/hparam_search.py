# Random search (25 configs) for PRESLEY in Barcelona using DEV validation.
# EarlyStopping + ModelCheckpoint monitoring "val_auc".
# Logs everything to TensorBoard (single root) + stores config.json + leaderboard.csv.

from __future__ import annotations

import sys
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import utils
from src.callbacks import EmissionsTrackerCallback 
from src.datamodule import ImageAuthorshipDataModule


### TORCH / PL RUNTIME KNOBS ###
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision("high")


### CONFIG ###
@dataclass(frozen=True)
class TrialConfig:
    # Fixed
    city: str = "barcelona"
    model: str = "PRESLEY"

    # Search space (one trial picks one of each)
    lr: float = 1e-3
    dropout: float = 0.2
    d: int = 256

    # Data / training constants for the search
    batch_size: int = 8192
    max_user_images: int = 20
    workers: int = 0

    # MUST be False for HPO (needs DEV validation)
    use_train_val: bool = False
    no_validation: bool = False

    # DeepSets flag (keep fixed unless you also want to search it)
    ds_no_rho: bool = False

    # Trainer
    max_epochs: int = 75
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 0

    # Optional speed/debug knobs
    limit_users: Optional[int] = None
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None


### HELPERS ###
def ensure_repo_root() -> None:
    if not Path("src").exists():
        raise RuntimeError("No encuentro 'src/'. Ejecuta este script desde la raíz del repo.")


def safe_float(x) -> Optional[float]:
    """Safely cast callback_metrics values to python float."""
    if x is None:
        return None
    try:
        if isinstance(x, (float, int)):
            return float(x)
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return None


def make_run_id(cfg: TrialConfig, trial_idx: int, seed_trial: int) -> str:
    """Compact run_id encoding the trial hyperparameters (friendly for folders/TB)."""
    use_rho = int(not cfg.ds_no_rho)
    return (
        f"trial{trial_idx:03d}"
        f"__d{cfg.d}"
        f"__lr{cfg.lr:g}"
        f"__do{cfg.dropout:g}"
        f"__bs{cfg.batch_size}"
        f"__K{cfg.max_user_images}"
        f"__tv{int(cfg.use_train_val)}"
        f"__noval{int(cfg.no_validation)}"
        f"__rho{use_rho}"
        f"__seed{seed_trial}"
    )


def sample_grid_no_replacement(
    lrs: List[float],
    dropouts: List[float],
    dims: List[int],
    n_samples: int,
    rng_seed: int,
) -> List[Tuple[float, float, int]]:
    """
    Sample without replacement from the cartesian product grid.
    NOTE: `dims` are latent dimensions (d), not DeepSets.
    """
    grid = list(itertools.product(lrs, dropouts, dims))
    if n_samples > len(grid):
        raise ValueError(f"n_samples={n_samples} > grid_size={len(grid)}")
    rnd = random.Random(rng_seed)
    rnd.shuffle(grid)
    return grid[:n_samples]


def build_logger(tb_root: Path, cfg: TrialConfig, run_id: str) -> TensorBoardLogger:
    """
    Single TB root; organized as:
        tb_root/<city>/<model>/<run_id>/events...
    """
    return TensorBoardLogger(
        save_dir=str(tb_root),
        name=f"{cfg.city}/{cfg.model}",
        version=run_id,
        default_hp_metric=False,
    )


def build_trainer(
    run_dir: Path,
    logger: TensorBoardLogger,
    cfg: TrialConfig
) -> Tuple[pl.Trainer, ModelCheckpoint]:
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    ckpt_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_auc",  # must match your logging key exactly
        mode="max",
        dirpath=str(run_dir / "checkpoints"),
        filename="best",
        save_on_train_epoch_end=False,
    )

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        min_delta=cfg.min_delta,
        patience=cfg.patience,
        check_on_train_epoch_end=False,
    )

    # Keep "last" checkpoint for debugging/repro
    ckpt_last = ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        dirpath=str(run_dir / "checkpoints"),
        save_on_train_epoch_end=True,
    )

    callbacks = [
        ckpt_best,
        early_stop,
        ckpt_last,
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer_kwargs = dict(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_model_summary=True,
        num_sanity_val_steps=2,
        deterministic=False,
    )

    if cfg.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = cfg.limit_train_batches
    if cfg.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = cfg.limit_val_batches

    return pl.Trainer(**trainer_kwargs), ckpt_best


def log_hparams_to_tensorboard(trainer: pl.Trainer, hparams: Dict, metrics: Dict) -> None:
    """
    TensorBoard HParams helper.
    Only logs scalar-friendly values (casts tensors/metrics when possible).
    """
    if trainer.logger is None or not hasattr(trainer.logger, "log_hyperparams"):
        return

    hp_clean = {}
    for k, v in hparams.items():
        if v is None:
            continue
        if isinstance(v, (int, float, str, bool)):
            hp_clean[k] = v
        else:
            hp_clean[k] = str(v)

    met_clean = {}
    for k, v in metrics.items():
        fv = safe_float(v)
        if fv is not None:
            met_clean[k] = fv

    trainer.logger.log_hyperparams(hp_clean, met_clean)


### TRIAL RUNNER ###
def run_one_trial(
    cfg: TrialConfig,
    trial_idx: int,
    out_root: Path,
    tb_root: Path,
) -> Dict:
    # Trial-level seed: reproducible + reduces randomness when comparing configs
    seed_trial = cfg.seed + trial_idx
    pl.seed_everything(seed_trial, workers=True)

    run_id = make_run_id(cfg, trial_idx, seed_trial)

    run_dir = out_root / cfg.city / cfg.model / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist exact trial config (so each run is reproducible)
    cfg_dict = asdict(cfg) | {"trial_idx": trial_idx, "seed_trial": seed_trial, "run_id": run_id}
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    # DataModule: with use_train_val=False, validation comes from DEV in your setup()
    dm = ImageAuthorshipDataModule(
        city=cfg.city,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        dataset_class=utils.get_dataset_constructor(cfg.model),
        use_train_val=cfg.use_train_val,
        limit_users=cfg.limit_users,
        model_name=cfg.model,
        max_user_images=cfg.max_user_images,
        no_validation=cfg.no_validation,
    )
    dm.setup()

    # Build model through the repo's factory (keeps signatures consistent)
    model_args = dict(
        d=cfg.d,
        lr=cfg.lr,
        dropout=cfg.dropout,
        max_user_images=cfg.max_user_images,
        ds_no_rho=cfg.ds_no_rho,
    )
    model = utils.get_model(cfg.model, model_args, dm.nusers)

    logger = build_logger(tb_root=tb_root, cfg=cfg, run_id=run_id)
    trainer, ckpt_best = build_trainer(run_dir=run_dir, logger=logger, cfg=cfg)

    t0 = time.time()
    trainer.fit(model=model, datamodule=dm)
    elapsed = time.time() - t0

    metrics = trainer.callback_metrics

    # Known logged keys in this codebase
    val_auc = safe_float(metrics.get("val_auc"))
    val_recall = safe_float(metrics.get("val_recall"))
    train_loss = safe_float(metrics.get("train_loss"))

    # Optional: captured if you later log it (script stays forward-compatible)
    val_ndcg = safe_float(metrics.get("val_ndcg")) or safe_float(metrics.get("val_mndcg"))

    best_score = safe_float(ckpt_best.best_model_score)
    best_path = ckpt_best.best_model_path if ckpt_best.best_model_path else None

    # Log HParams summary to TensorBoard
    hparams_for_tb = {
        "d": cfg.d,
        "lr": cfg.lr,
        "dropout": cfg.dropout,
        "max_user_images": cfg.max_user_images,
        "batch_size": cfg.batch_size,
        "use_train_val": cfg.use_train_val,
        "ds_no_rho": cfg.ds_no_rho,
        "seed_trial": seed_trial,
    }
    metrics_for_tb = {
        "hp/val_auc": val_auc,
        "hp/val_recall": val_recall,
        "hp/val_ndcg": val_ndcg,
        "hp/train_loss": train_loss,
        "hp/best_model_score": best_score,
    }
    log_hparams_to_tensorboard(trainer, hparams_for_tb, metrics_for_tb)

    # Row for leaderboard.csv
    row = dict(
        trial_idx=trial_idx,
        run_id=run_id,
        city=cfg.city,
        model=cfg.model,
        d=cfg.d,
        lr=cfg.lr,
        dropout=cfg.dropout,
        max_user_images=cfg.max_user_images,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        use_train_val=int(cfg.use_train_val),
        no_validation=int(cfg.no_validation),
        ds_no_rho=int(cfg.ds_no_rho),
        seed_base=cfg.seed,
        seed_trial=seed_trial,
        elapsed_sec=elapsed,
        train_loss=train_loss,
        val_auc=val_auc,
        val_recall=val_recall,
        val_ndcg=val_ndcg,
        best_model_score=best_score,
        best_ckpt_path=best_path,
        run_dir=str(run_dir),
        tb_dir=str(Path(tb_root) / cfg.city / cfg.model / run_id),
    )
    return row


def append_row_csv(csv_path: Path, row: Dict) -> None:
    df_row = pd.DataFrame([row])
    if not csv_path.exists():
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)


### MAIN ###
def main() -> None:
    ensure_repo_root()

    # Search space (exact grid you defined)
    lrs = [1e-3, 1e-4, 1e-5]
    dropouts = [0.2, 0.4, 0.6, 0.8]
    dims = [16, 64, 256, 1024]  # latent dims (d)
    n_trials = 25

    project_out = Path("optimization")
    out_root = project_out / "runs"
    tb_root = project_out / "tb"
    out_root.mkdir(parents=True, exist_ok=True)
    tb_root.mkdir(parents=True, exist_ok=True)

    leaderboard_csv = project_out / "leaderboard.csv"

    base_seed = 0

    combos = sample_grid_no_replacement(
        lrs=lrs, dropouts=dropouts, dims=dims, n_samples=n_trials, rng_seed=base_seed
    )

    # Skip already completed trials if leaderboard exists
    done_run_ids = set()
    if leaderboard_csv.exists():
        try:
            df_done = pd.read_csv(leaderboard_csv)
            if "run_id" in df_done.columns:
                done_run_ids = set(df_done["run_id"].astype(str).tolist())
        except Exception:
            done_run_ids = set()

    # Base config for the sweep
    base_cfg = TrialConfig(
        city="barcelona",
        model="PRESLEY",
        batch_size=8192,
        max_user_images=20,
        workers=0,
        use_train_val=False,
        no_validation=False,
        ds_no_rho=False,
        max_epochs=75,
        patience=10,
        min_delta=1e-4,
        seed=base_seed,
        limit_users=None,
        limit_train_batches=None,
        limit_val_batches=None,
    )

    print(f"[HPO] output folder: {project_out.resolve()}")
    print(f"[HPO] runs: {out_root.resolve()}")
    print(f"[HPO] tb: {tb_root.resolve()}")
    print(f"[HPO] leaderboard: {leaderboard_csv.resolve()}")
    print(f"[HPO] trials: {n_trials} (grid_size={len(lrs)*len(dropouts)*len(dims)})")
    print("[HPO] monitor: val_auc (mode=max)")

    for i, (lr, dropout, d) in enumerate(combos):
        cfg = TrialConfig(**(asdict(base_cfg) | {"lr": lr, "dropout": dropout, "d": d}))

        seed_trial = cfg.seed + i
        run_id = make_run_id(cfg, i, seed_trial)

        if run_id in done_run_ids:
            print(f"[HPO][SKIP] {run_id} (ya está en leaderboard)")
            continue

        print(f"[HPO][TRIAL {i+1}/{n_trials}] d={d} lr={lr} dropout={dropout} run_id={run_id}")

        try:
            row = run_one_trial(cfg=cfg, trial_idx=i, out_root=out_root, tb_root=tb_root)
            append_row_csv(leaderboard_csv, row)

            # Provisional top: val_auc first, then recall, then ndcg (if present)
            df = pd.read_csv(leaderboard_csv)
            df_sorted = df.sort_values(
                by=["val_auc", "val_recall", "val_ndcg"],
                ascending=[False, False, False],
                na_position="last",
            )
            best = df_sorted.iloc[0].to_dict()
            print(
                f"[HPO][BEST SO FAR] val_auc={best.get('val_auc')} "
                f"val_recall={best.get('val_recall')} val_ndcg={best.get('val_ndcg')} "
                f"run_id={best.get('run_id')}"
            )

        except Exception as e:
            # Don't let a broken trial crash the whole sweep
            err_path = out_root / base_cfg.city / base_cfg.model / run_id / "error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(repr(e))
            print(f"[HPO][ERROR] Trial falló: {e}")
            continue

    # Final sort of leaderboard
    if leaderboard_csv.exists():
        df = pd.read_csv(leaderboard_csv)
        df_sorted = df.sort_values(
            by=["val_auc", "val_recall", "val_ndcg"],
            ascending=[False, False, False],
            na_position="last",
        )
        df_sorted.to_csv(leaderboard_csv, index=False)

        best = df_sorted.iloc[0].to_dict()
        print("[HPO][DONE] Best config:")
        print(
            f"  d={best.get('d')} lr={best.get('lr')} dropout={best.get('dropout')}\n"
            f"  val_auc={best.get('val_auc')} val_recall={best.get('val_recall')} val_ndcg={best.get('val_ndcg')}\n"
            f"  best_ckpt_path={best.get('best_ckpt_path')}\n"
            f"  run_dir={best.get('run_dir')}\n"
            f"  tb_dir={best.get('tb_dir')}"
        )


if __name__ == "__main__":
    main()
