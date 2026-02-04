from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from codecarbon import EmissionsTracker


CITIES: List[str] = ["gijon", "barcelona", "madrid", "newyork", "paris", "london"]
MODEL = "PRESLEY"


def _ensure_repo_root() -> None:
    if not Path("main.py").exists():
        raise RuntimeError("No encuentro main.py. Ejecuta este script desde la raíz del repo.")
    if not Path("src").exists():
        raise RuntimeError("No encuentro src/. Ejecuta este script desde la raíz del repo.")
    if not Path("results").exists():
        raise RuntimeError("No encuentro results/. ¿Tienes checkpoints ya generados?")


def _find_latest_run_ckpt(city: str, must_contain: Optional[List[str]] = None) -> Tuple[Path, Path]:
    base = Path("results") / city / MODEL
    if not base.exists():
        raise FileNotFoundError(f"No existe {base}")

    cand = []
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        if must_contain and not all(tok in run_dir.name for tok in must_contain):
            continue
        ckpt = run_dir / "checkpoints" / "last.ckpt"
        if ckpt.exists():
            cand.append((run_dir.stat().st_mtime, run_dir, ckpt))

    if not cand:
        msg = f"No encontré last.ckpt en {base}"
        if must_contain:
            msg += f" (filtrando por tokens: {must_contain})"
        raise FileNotFoundError(msg)

    cand.sort(key=lambda x: x[0], reverse=True)
    _, run_dir, ckpt = cand[0]
    return run_dir, ckpt


def _load_run_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No encuentro config.json en {cfg_path} (necesario para replicar args del ckpt).")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def _pick(cfg: Dict, keys: List[str], required: bool = True):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    if required:
        raise KeyError(f"Config no contiene ninguna de estas keys: {keys}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_label", required=True, type=str, help="Etiqueta a escribir en el CSV (ej: BRIE o BRIE+DeepSets)")
    ap.add_argument("--out_csv", default="results/eff_inference_test.csv", type=str)
    ap.add_argument("--filter_tokens", default="tv1,noval1", type=str,
                    help="Tokens requeridos en el nombre del run_dir, separados por coma. Usa '' para desactivar.")
    ap.add_argument("--seed", default="", type=str, help="Si lo pones, añade filtro seedX (ej: 0 o 1).")
    ap.add_argument("--extra_args", default="", type=str,
                    help="Args extra literales para main.py stage=test (solo si no están en config.json).")
    args = ap.parse_args()

    _ensure_repo_root()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tokens = [t.strip() for t in args.filter_tokens.split(",") if t.strip()]
    if args.seed.strip():
        tokens.append(f"seed{args.seed.strip()}")

    extra_args_list = args.extra_args.split() if args.extra_args.strip() else []
    python_exe = sys.executable

    rows = []

    for city in CITIES:
        run_dir, ckpt = _find_latest_run_ckpt(city, must_contain=tokens if tokens else None)
        cfg = _load_run_config(run_dir)

        # intentamos ser robustos con nombres de keys
        d = int(_pick(cfg, ["d", "latent_dim", "embedding_dim"]))
        lr = _pick(cfg, ["lr", "learning_rate"])
        dropout = _pick(cfg, ["dropout", "p_dropout"], required=False)
        batch_size = int(_pick(cfg, ["batch_size", "bs"]))
        K = int(_pick(cfg, ["max_user_images", "K", "k"], required=False) or 20)

        # flag deepsets opcional
        ds_no_rho = _pick(cfg, ["ds_no_rho"], required=False)
        ds_no_rho = bool(int(ds_no_rho)) if ds_no_rho is not None else False

        cmd = [
            python_exe, "main.py",
            "--city", city,
            "--stage", "test",
            "--model", MODEL,
            "-d", str(d),
            "--lr", str(lr),
            "--batch_size", str(batch_size),
            "--max_user_images", str(K),
            "--use_train_val",
            "--no_validation",
            "--workers", "0",
            "--ckpt_path", str(ckpt),
        ]

        # solo pasamos dropout si existe en config
        if dropout is not None:
            cmd += ["--dropout", str(dropout)]

        # solo pasamos ds_no_rho si aplica
        if ds_no_rho:
            cmd += ["--ds_no_rho"]

        # extra args manuales (por si algo no está en config)
        if extra_args_list:
            cmd += extra_args_list

        print("\n" + "=" * 120)
        print(f"[MEASURE] city={city} | label={args.model_label}")
        print("  run_dir:", run_dir)
        print("  ckpt:   ", ckpt)
        print("  parsed:", f"d={d} lr={lr} do={dropout} bs={batch_size} K={K} ds_no_rho={ds_no_rho}")
        print("  cmd:    ", " ".join(cmd))
        print("=" * 120)

        tracker = EmissionsTracker(
            project_name=f"inference_{args.model_label}",
            output_dir=str(out_csv.parent),
            output_file=f"codecarbon_inference_{args.model_label}_{city}.csv",
            log_level="error",
            allow_multiple_runs=True,
        )

        t0 = time.perf_counter()
        tracker.start()
        proc = subprocess.run(cmd, check=False)
        emissions_kg = tracker.stop()
        t1 = time.perf_counter()

        if proc.returncode != 0:
            raise RuntimeError(f"Falló stage=test en {city} (returncode={proc.returncode}).")

        rows.append({
            "city": city,
            "label": args.model_label,
            "d": d,
            "lr": str(lr),
            "dropout": None if dropout is None else float(dropout),
            "batch_size": batch_size,
            "K": K,
            "ds_no_rho": int(ds_no_rho),
            "inference_time_s": float(t1 - t0),
            "emissions_kg": None if emissions_kg is None else float(emissions_kg),
            "run_dir": str(run_dir),
            "ckpt_path": str(ckpt),
        })

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[OK] guardado parcial: {out_csv}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[DONE] guardado final: {out_csv}")


if __name__ == "__main__":
    main()
