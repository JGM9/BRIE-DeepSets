import pickle
from pathlib import Path
import pandas as pd

# Path to TRAIN_IMG file (relative to project root)
TRAIN_IMG_PATH = Path("data/barcelona/data_10+10/TRAIN_IMG")

# Load dataframe
with open(TRAIN_IMG_PATH, "rb") as f:
    df = pickle.load(f)

# Count number of images per user
images_per_user = df.groupby("id_user")["id_img"].count()

# Basic statistics
print("Images per user statistics:")
print(images_per_user.describe())

# Percentage of users above different thresholds
thresholds = [5, 10, 15, 20, 30, 50]

print("\nPercentage of users with more than K images:")
for k in thresholds:
    percentage = (images_per_user > k).mean() * 100
    print(f"K > {k}: {percentage:.2f}%")
