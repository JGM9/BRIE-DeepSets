# Train and evaluate the best HPO configuration across all cities.
# The best trial is selected from optimization/leaderboard.csv using val_auc.
# For each city, a final training is performed with use_train_val + no_validation
# (no early stopping), followed by automatic test using the generated last.ckpt.
# Final test metrics are aggregated into results/final_test_results.csv.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


### SETTINGS ###
CITIES: List[str] = ["barcelona", "madrid", "gijon", "paris", "london", "newyork"]
MODEL = "PRESLEY"

FINAL_MAX_EPOCHS = 75
SEED = 0
WORKERS = 0

LEADERBOARD = Path("optimization") / "leaderboard.csv"
OUT_CSV = "results/final_test_results.csv"


### HELPERS ###
def _ensure_repo_root() -> None:
    if not Path("main.py").exists():
        raise RuntimeError("No encuentro main.py. Ejecuta este script desde la raíz del repo.")
    if not Path("src").exists():
        raise RuntimeError("No encuentro src/. Ejecuta este script desde la raíz del repo.")


def _run(cmd: List[str]) -> None:
    """Run a command and raise if it fails."""
    print("\n[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando falló (returncode={proc.returncode}).")


def _load_best_row() -> Dict:
    if not LEADERBOARD.exists():
        raise FileNotFoundError(f"No existe {LEADERBOARD}. Primero ejecuta el HPO.")

    df = pd.read_csv(LEADERBOARD)

    if "val_auc" not in df.columns:
        raise RuntimeError("El leaderboard no tiene columna 'val_auc' (necesaria para elegir el mejor).")

    # NOTE: push failed rows (NaN) to the end
    df = df.sort_values(by=["val_auc"], ascending=False, na_position="last")

    best = df.iloc[0].to_dict()
    required = ["d", "lr", "dropout", "batch_size", "max_user_images"]
    for k in required:
        if k not in best or pd.isna(best[k]):
            raise RuntimeError(f"Falta '{k}' en la mejor fila del leaderboard.")

    return best


def _bool_from_any(x, default=False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    try:
        xi = int(x)
        return bool(xi)
    except Exception:
        s = str(x).strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n"}:
            return False
    return default


def _find_run_dir(
    city: str,
    d: str,
    lr: str,
    dropout: str,
    batch_size: str,
    K: str,
    rho: str,
    seed: str
) -> Path:
    """
    Find the FINAL training run_dir created by main.py (tv=1, noval=1).

    We avoid reconstructing run_id from floats (lr/dropout formatting can vary).
    However, if multiple matches exist, we *prefer* the one that also contains
    the literal lr/dropout strings (as stored in leaderboard).
    """
    base = Path("results") / city / MODEL
    if not base.exists():
        raise FileNotFoundError(f"No existe {base}. ¿Se llegó a entrenar algo para {city}?")

    must_have = [
        f"d{d}",
        f"bs{batch_size}",
        f"K{K}",
        "tv1",
        "noval1",
        f"rho{rho}",
        f"seed{seed}",
    ]

    candidates: List[Path] = []
    for p in base.iterdir():
        if p.is_dir() and all(tok in p.name for tok in must_have):
            candidates.append(p)

    if not candidates:
        # fallback: only by tv/noval/seed
        must_have2 = ["tv1", "noval1", f"seed{seed}"]
        for p in base.iterdir():
            if p.is_dir() and all(tok in p.name for tok in must_have2):
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No pude encontrar el run_dir final en {base}. "
            f"Esperaba algo con tv1/noval1/seed{seed}."
        )

    # Prefer candidates that also contain the literal lr/dropout tokens (if present)
    lr_tok = f"lr{lr}"
    do_tok = f"do{dropout}"
    preferred = [p for p in candidates if (lr_tok in p.name and do_tok in p.name)]
    if preferred:
        candidates = preferred

    # If still multiple, pick most recent
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_metrics_from_test_json(test_json_path: Path) -> Dict[str, Optional[float]]:
    """
    main.py stores a dict like:
      { "PRESLEY": { ... metrics ... }, "RANDOM": {...}, ... }

    We extract PRESLEY metrics and normalize keys to:
      - MRecall@10
      - MNDCG@10
      - MAUC
    """
    with open(test_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_block = data.get(MODEL, data)  # fallback if stored without wrapper

    key_map = {
        "MRecall@10": ["MRecall@10", "mrecall@10", "recall@10", "Recall@10"],
        "MNDCG@10": ["MNDCG@10", "mndcg@10", "ndcg@10", "NDCG@10"],
        "MAUC": ["MAUC", "mauc", "AUC", "auc"],
    }

    out: Dict[str, Optional[float]] = {}
    for out_key, candidates in key_map.items():
        val = None
        for k in candidates:
            if isinstance(model_block, dict) and k in model_block:
                val = model_block[k]
                break
        out[out_key] = None if val is None else float(val)
    return out


### MAIN ###
def main() -> None:
    _ensure_repo_root()
    best = _load_best_row()

    # IMPORTANT: keep as strings (avoid float formatting issues: 1e-03 vs 0.001)
    d = str(int(best["d"]))
    lr = str(best["lr"])
    dropout = str(best["dropout"])
    batch_size = str(int(best["batch_size"]))
    K = str(int(best["max_user_images"]))

    ds_no_rho = _bool_from_any(best.get("ds_no_rho", 0), default=False)
    rho = "0" if ds_no_rho else "1"
    seed = str(int(best.get("seed", SEED))) if "seed" in best and not pd.isna(best["seed"]) else str(SEED)

    print("[BEST FROM HPO]")
    print(f"  d={d} lr={lr} dropout={dropout} batch_size={batch_size} K={K} ds_no_rho={ds_no_rho} seed={seed}")
    print(f"  FINAL_MAX_EPOCHS={FINAL_MAX_EPOCHS}")
    print(f"  Cities: {CITIES}")

    python_exe = sys.executable
    rows = []

    for city in CITIES:
        ### 1) TRAIN FINAL ###
        train_cmd = [
            python_exe, "main.py",
            "--city", city,
            "--stage", "train",
            "--model", MODEL,
            "-d", d,
            "--lr", lr,
            "--dropout", dropout,
            "--batch_size", batch_size,
            "--max_epochs", str(FINAL_MAX_EPOCHS),
            "--max_user_images", K,
            "--use_train_val",
            "--no_validation",
            "--workers", str(WORKERS),
            "--seed", seed,
        ]
        if ds_no_rho:
            train_cmd.append("--ds_no_rho")

        _run(train_cmd)

        # Find run_dir + last checkpoint without rebuilding run_id manually
        run_dir = _find_run_dir(
            city=city, d=d, lr=lr, dropout=dropout, batch_size=batch_size, K=K, rho=rho, seed=seed
        )
        ckpt_path = run_dir / "checkpoints" / "last.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No encuentro last.ckpt en: {ckpt_path}")

        ### 2) TEST ###
        test_cmd = [
            python_exe, "main.py",
            "--city", city,
            "--stage", "test",
            "--model", MODEL,
            "-d", d,
            "--lr", lr,
            "--dropout", dropout,
            "--batch_size", batch_size,
            "--max_user_images", K,
            "--use_train_val",
            "--no_validation",
            "--workers", str(WORKERS),
            "--seed", seed,
            "--ckpt_path", str(ckpt_path),
        ]
        if ds_no_rho:
            test_cmd.append("--ds_no_rho")

        _run(test_cmd)

        ### 3) PARSE METRICS ###
        test_json = run_dir / "test_metrics.json"
        if not test_json.exists():
            raise FileNotFoundError(f"No encuentro test_metrics.json en: {test_json}")

        m = _extract_metrics_from_test_json(test_json)

        row = {
            "city": city,
            "d": int(d),
            "lr": lr,
            "dropout": dropout,
            "batch_size": int(batch_size),
            "K": int(K),
            "ds_no_rho": int(ds_no_rho),
            "seed": int(seed),
            "ckpt_path": str(ckpt_path),
            "run_dir": str(run_dir),
            **m,
        }
        rows.append(row)

        # incremental save (so you can stop mid-way without losing everything)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"[OK] guardado parcial: {OUT_CSV}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n[DONE] Resultados finales guardados en:", OUT_CSV)


if __name__ == "__main__":
    main()
