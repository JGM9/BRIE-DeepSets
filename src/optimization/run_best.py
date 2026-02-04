# 1) Lee el mejor trial del HPO (optimization/leaderboard.csv) por val_auc
# 2) Para cada ciudad: entrena FINAL con use_train_val + no_validation (sin early stopping)
# 3) Ejecuta test automático usando el last.ckpt del run final
# 4) Agrega resultados a results/final_test_results.csv

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# =========================
# Ajustes
# =========================
CITIES: List[str] = ["barcelona", "madrid", "gijon", "paris", "london", "newyork"]
MODEL = "PRESLEY"

FINAL_MAX_EPOCHS = 75
SEED = 0
WORKERS = 0

LEADERBOARD = Path("optimization") / "leaderboard.csv"
OUT_CSV = "results/final_test_results.csv"


def _ensure_repo_root() -> None:
    if not Path("main.py").exists():
        raise RuntimeError("No encuentro main.py. Ejecuta este script desde la raíz del repo.")
    if not Path("src").exists():
        raise RuntimeError("No encuentro src/. Ejecuta este script desde la raíz del repo.")


def _run(cmd: List[str]) -> None:
    print("\n" + "=" * 110)
    print("[RUN]", " ".join(cmd))
    print("=" * 110)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando falló (returncode={proc.returncode}).")


def _load_best_row() -> Dict:
    if not LEADERBOARD.exists():
        raise FileNotFoundError(f"No existe {LEADERBOARD}. Primero ejecuta el HPO.")

    df = pd.read_csv(LEADERBOARD)

    if "val_auc" not in df.columns:
        raise RuntimeError("El leaderboard no tiene columna 'val_auc' (necesaria para elegir el mejor).")

    # OJO: si hay filas fallidas con NaN, las mandamos al final
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


def _find_run_dir(city: str, d: str, lr: str, dropout: str, batch_size: str, K: str, rho: str, seed: str) -> Path:
    """
    Encuentra el run_dir creado por main.py para el entrenamiento FINAL (tv=1, noval=1)
    sin depender del formato exacto de floats (buscando por partes).
    """
    base = Path("results") / city / MODEL
    if not base.exists():
        raise FileNotFoundError(f"No existe {base}. ¿Se llegó a entrenar algo para {city}?")

    # main.py construye run_id con tokens:
    # d{d}__lr{lr}__do{dropout}__bs{batch_size}__K{K}__tv1__noval1__rho{rho}__seed{seed}
    # Pero lr/dropout pueden tener formatos distintos. Buscamos por todos los tokens fijos y algunos variables.
    must_have = [
        f"d{d}",
        f"bs{batch_size}",
        f"K{K}",
        "tv1",
        "noval1",
        f"rho{rho}",
        f"seed{seed}",
    ]

    candidates = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if all(tok in name for tok in must_have):
            # además, intentamos que también contenga lr y dropout tal cual vienen del best row (si coincide)
            # pero sin exigirlo.
            candidates.append(p)

    if not candidates:
        # fallback menos estricto: solo por tv/noval/seed
        must_have2 = ["tv1", "noval1", f"seed{seed}"]
        for p in base.iterdir():
            if p.is_dir() and all(tok in p.name for tok in must_have2):
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No pude encontrar el run_dir final en {base}. "
            f"Esperaba algo con tv1/noval1/seed{seed}."
        )

    # si hay más de uno, nos quedamos con el más reciente
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_metrics_from_test_json(test_json_path: Path) -> Dict[str, Optional[float]]:
    """
    main.py guarda un dict tipo:
      { "PRESLEY": { ... métricas ... }, "RANDOM": {...}, ... }
    Queremos PRESLEY y extraer MRecall@10, MNDCG@10, MAUC (si existen).
    """
    with open(test_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_block = data.get(MODEL, data)  # por si un día guardas directo sin wrapper

    # Nombres posibles (por si el test usa otras keys)
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


def main() -> None:
    _ensure_repo_root()
    best = _load_best_row()

    # IMPORTANTÍSIMO: NO convierto a float para recomponer strings (evita líos con 1e-03 vs 0.001)
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
        # -----------------------
        # 1) TRAIN FINAL
        # -----------------------
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

        # localizar run_dir y checkpoint last.ckpt sin reconstruir run_id a mano
        run_dir = _find_run_dir(city=city, d=d, lr=lr, dropout=dropout, batch_size=batch_size, K=K, rho=rho, seed=seed)
        ckpt_path = run_dir / "checkpoints" / "last.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No encuentro last.ckpt en: {ckpt_path}")

        # -----------------------
        # 2) TEST
        # -----------------------
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

        # -----------------------
        # 3) Parse metrics
        # -----------------------
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

        # guardar progreso incremental por si paras a mitad
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"[OK] guardado parcial: {OUT_CSV}")

    # final
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n[DONE] Resultados finales guardados en:", OUT_CSV)


if __name__ == "__main__":
    main()
