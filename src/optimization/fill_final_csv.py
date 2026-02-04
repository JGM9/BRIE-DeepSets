from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

MODEL = "PRESLEY"
RUNS_ROOT = Path("results")
CSV_PATH = Path("results") / "final_test_results.csv"


def _ensure_repo_root() -> None:
    if not Path("main.py").exists():
        raise RuntimeError("Ejecuta el script desde la raíz del repo (no encuentro main.py).")
    if not RUNS_ROOT.exists():
        raise RuntimeError("No existe la carpeta results/.")


def _find_latest_run(city: str) -> Optional[Path]:
    base = RUNS_ROOT / city / MODEL
    if not base.exists():
        return None

    results = [p for p in base.iterdir() if p.is_dir()]
    if not results:
        return None

    # Preferimos train+val sin validación (tv1, noval1)
    preferred = [p for p in results if "tv1" in p.name and "noval1" in p.name]
    candidates = preferred if preferred else results

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_metrics(test_json: Path) -> Dict[str, Optional[float]]:
    with test_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if MODEL not in data:
        raise RuntimeError(f"{test_json} no contiene bloque '{MODEL}'.")

    m = data[MODEL]

    return {
        "MAUC": float(m["auc_all"]) if "auc_all" in m else None,
        "MRecall@10": float(m["recall@10_10plus"]) if "recall@10_10plus" in m else None,
        "MNDCG@10": float(m["ndcg@10_10plus"]) if "ndcg@10_10plus" in m else None,
    }


def main() -> None:
    _ensure_repo_root()

    if not CSV_PATH.exists():
        raise RuntimeError(f"No existe {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    for col in ["MRecall@10", "MNDCG@10", "MAUC"]:
        if col not in df.columns:
            df[col] = None

    updated = 0
    skipped = []

    for i, row in df.iterrows():
        city = str(row["city"]).lower()

        run_dir = _find_latest_run(city)
        if run_dir is None:
            skipped.append(city)
            continue

        test_json = run_dir / "test_metrics.json"
        if not test_json.exists():
            skipped.append(city)
            continue

        metrics = _extract_metrics(test_json)

        for k, v in metrics.items():
            df.at[i, k] = v

        updated += 1

    df.to_csv(CSV_PATH, index=False)

    print(f"[OK] Filas actualizadas: {updated}")
    if skipped:
        print("[WARN] Ciudades sin resultados:", ", ".join(skipped))
    print(f"[OK] CSV final listo en: {CSV_PATH}")


if __name__ == "__main__":
    main()
