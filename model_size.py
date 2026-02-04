from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# Datos (de tus logs)
# =========================
models = ["BRIE", "BRIE+Deep Sets"]

trainable_params = [2_200_000, 209_000]   # 2.2 M y 209 K
estimated_mb = [8.979, 0.838]            # estimación Lightning

# Colores EXACTOS (mismo tono que el resto de figuras)
colors = ["#0000FF", "#FF0000"]  # azul, rojo

out_path = "eff_model_size.png"

# =========================
# Plot
# =========================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel 1: parámetros entrenables
axes[0].bar(models, trainable_params, color=colors)
axes[0].set_title("Parámetros entrenables")
axes[0].set_ylabel("Número de parámetros")
axes[0].ticklabel_format(style="plain", axis="y")
axes[0].grid(axis="y", alpha=0.3)

# Panel 2: tamaño estimado
axes[1].bar(models, estimated_mb, color=colors)
axes[1].set_title("Tamaño estimado del modelo")
axes[1].set_ylabel("MB (estimación Lightning)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"[OK] Figura guardada en: {out_path}")
