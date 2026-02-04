import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
RESULTS_DIR = Path("results")
OUT_CO2 = RESULTS_DIR / "inference_co2_compare.png"
OUT_TIME = RESULTS_DIR / "inference_time_compare.png"

# Colores pedidos
COLOR_BRIE = "blue"
COLOR_DS = "red"


# ----------------------------
# Helpers
# ----------------------------
def _fail(msg: str, code: int = 1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def _pick_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_csv(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        _fail(f"No encuentro el CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        _fail(f"CSV vacío: {path}")
    if "city" not in df.columns:
        _fail(f"El CSV {path} no tiene columna 'city'. Columnas: {list(df.columns)}")
    # Normaliza city a string limpio
    df["city"] = df["city"].astype(str).str.strip().str.lower()
    return df


def co2_to_g(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve emisiones en gCO2 a partir de posibles columnas:
    - emissions_kg (kgCO2)
    - emissions_g (gCO2)
    - emissions (si es kg o g no se sabe: intentamos inferir)
    """
    cols = df.columns

    if "emissions_kg" in cols:
        s = pd.to_numeric(df["emissions_kg"], errors="coerce") * 1000.0
        return s

    if "emissions_g" in cols:
        s = pd.to_numeric(df["emissions_g"], errors="coerce")
        return s

    if "emissions" in cols:
        s = pd.to_numeric(df["emissions"], errors="coerce")
        # Heurística mínima: si la mediana es < 1, probablemente está en kg -> pasar a g
        med = float(s.median(skipna=True)) if s.notna().any() else float("nan")
        if med == med and med < 1.0:
            return s * 1000.0
        return s

    _fail(
        "No encuentro columna de emisiones. Esperaba 'emissions_kg' o 'emissions_g' o 'emissions'. "
        f"Columnas: {list(cols)}"
    )


def time_to_s(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve tiempo de inferencia en segundos, buscando nombres típicos.
    """
    cols = df.columns
    for c in ["inference_time_s", "inference_time", "time_s", "seconds"]:
        if c in cols:
            return pd.to_numeric(df[c], errors="coerce")
    _fail(
        "No encuentro columna de tiempo de inferencia. Esperaba 'inference_time_s' (u otras variantes). "
        f"Columnas: {list(cols)}"
    )


def label_or_default(df: pd.DataFrame, default: str) -> str:
    if "label" in df.columns and df["label"].notna().any():
        # Ojo: si hay más de un label en el CSV, eso es sospechoso; tomamos el más frecuente.
        return str(df["label"].mode(dropna=True).iloc[0])
    return default


def validate_one_row_per_city(df: pd.DataFrame, name: str):
    dup = df["city"].duplicated(keep=False)
    if dup.any():
        dups = df.loc[dup, "city"].value_counts().to_dict()
        _fail(
            f"En {name} hay ciudades repetidas (debería ser 1 fila por ciudad para estas gráficas). "
            f"Repeticiones: {dups}"
        )


def align_cities(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str):
    cities_a = set(df_a["city"].tolist())
    cities_b = set(df_b["city"].tolist())
    common = sorted(list(cities_a & cities_b))
    if not common:
        _fail(f"No hay intersección de ciudades entre {name_a} y {name_b}.")
    missing_a = sorted(list(cities_b - cities_a))
    missing_b = sorted(list(cities_a - cities_b))
    if missing_a:
        print(f"[WARN] {name_a} no tiene estas ciudades (se ignorarán): {missing_a}")
    if missing_b:
        print(f"[WARN] {name_b} no tiene estas ciudades (se ignorarán): {missing_b}")
    df_a2 = df_a[df_a["city"].isin(common)].copy()
    df_b2 = df_b[df_b["city"].isin(common)].copy()
    # Orden consistente
    df_a2 = df_a2.set_index("city").loc[common].reset_index()
    df_b2 = df_b2.set_index("city").loc[common].reset_index()
    return common, df_a2, df_b2


def plot_bars(cities, y1, y2, label1, label2, ylabel, title, outpath: Path):
    x = list(range(len(cities)))
    width = 0.38

    plt.figure(figsize=(11, 4.8))
    plt.bar([i - width / 2 for i in x], y1, width=width, label=label1, color=COLOR_BRIE)
    plt.bar([i + width / 2 for i in x], y2, width=width, label=label2, color=COLOR_DS)

    plt.xticks(x, cities)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[OK] Guardado: {outpath}")


# ----------------------------
# Main
# ----------------------------
def main():
    if not RESULTS_DIR.exists():
        _fail(f"No existe la carpeta {RESULTS_DIR.resolve()}")

    # Por defecto busco nombres típicos; si no existen, obliga a pasar rutas por CLI.
    # Uso:
    #   python plot_inference_compare.py results/inference_brie_original.csv results/eff_inference_test.csv
    arg_paths = [Path(a) for a in sys.argv[1:3]]  # hasta 2 args

    if len(arg_paths) == 2:
        brie_csv = arg_paths[0]
        ds_csv = arg_paths[1]
    else:
        brie_csv = _pick_first_existing(
            [
                RESULTS_DIR / "inference_brie_original.csv",
                RESULTS_DIR / "eff_inference_original.csv",
                RESULTS_DIR / "efficiency_original.csv",
            ]
        )
        ds_csv = _pick_first_existing(
            [
                RESULTS_DIR / "eff_inference_test.csv",
                RESULTS_DIR / "eff_inference_deepsets.csv",
                RESULTS_DIR / "inference_deepsets.csv",
            ]
        )

    if brie_csv is None or ds_csv is None:
        _fail(
            "No pude autodetectar los dos CSV en results/. "
            "Ejecuta pasando rutas explícitas:\n"
            "  python plot_inference_compare.py <csv_brie> <csv_brie_deepsets>"
        )

    df_brie = load_csv(brie_csv)
    df_ds = load_csv(ds_csv)

    # Estas gráficas asumen 1 fila por ciudad (como tu CSV de DeepSets).
    validate_one_row_per_city(df_brie, f"BRIE ({brie_csv.name})")
    validate_one_row_per_city(df_ds, f"BRIE+DeepSets ({ds_csv.name})")

    brie_label = label_or_default(df_brie, "BRIE")
    ds_label = label_or_default(df_ds, "BRIE+DeepSets")

    cities, df_brie, df_ds = align_cities(df_brie, df_ds, "BRIE", "BRIE+DeepSets")

    brie_time = time_to_s(df_brie)
    ds_time = time_to_s(df_ds)

    brie_co2_g = co2_to_g(df_brie)
    ds_co2_g = co2_to_g(df_ds)

    # Checks de NaNs (si hay NaNs, es que el parseo falló o hay celdas vacías)
    if brie_time.isna().any() or ds_time.isna().any():
        _fail("Hay NaNs en tiempos de inferencia tras parsear. Revisa columnas/valores en tus CSV.")
    if brie_co2_g.isna().any() or ds_co2_g.isna().any():
        _fail("Hay NaNs en emisiones tras parsear. Revisa columnas/valores en tus CSV.")

    plot_bars(
        cities=cities,
        y1=brie_co2_g.tolist(),
        y2=ds_co2_g.tolist(),
        label1="BRIE",
        label2="BRIE + DeepSets",
        ylabel="Emissions (gCO2)",
        title="Inference CO2 emissions",
        outpath=OUT_CO2,
    )

    plot_bars(
        cities=cities,
        y1=brie_time.tolist(),
        y2=ds_time.tolist(),
        label1="BRIE",
        label2="BRIE + DeepSets",
        ylabel="Time (s)",
        title="Inference Times",
        outpath=OUT_TIME,
    )

    # Mini resumen por consola (útil para detectar cosas raras rápido)
    print("\n[SUMMARY]")
    for c, t1, t2, e1, e2 in zip(cities, brie_time, ds_time, brie_co2_g, ds_co2_g):
        print(f"- {c:10s}  time: {t1:8.3f}s vs {t2:8.3f}s   |   CO2: {e1:8.5f}g vs {e2:8.5f}g")


if __name__ == "__main__":
    main()
