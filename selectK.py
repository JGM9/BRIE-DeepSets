import pickle
import pandas as pd

# Ruta a tu archivo TRAIN_IMG
ruta = r"C:\Users\Usuario\Desktop\BRIE-DeepSets\data\barcelona\data_10+10\TRAIN_IMG"

# Cargar el dataframe
df = pickle.load(open(ruta, "rb"))

# Agrupar por usuario y contar cuántas imágenes tiene cada uno
imagenes_por_usuario = df.groupby("id_user")["id_img"].count()

# Mostrar estadísticas básicas
print(imagenes_por_usuario.describe())

# Opcional: ver cuántos usuarios tienen más de X imágenes
for limite in [5, 10, 15, 20, 30, 50]:
    porcentaje = (imagenes_por_usuario > limite).mean() * 100
    print(f"Usuarios con más de {limite} imágenes: {porcentaje:.2f}%")
