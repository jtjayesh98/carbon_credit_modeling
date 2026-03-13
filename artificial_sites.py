import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "images"

predict_year = "2010_15"
ex_ante = True
data_preprocessing = "smote"   # must match prediction

if ex_ante:
    train_year = (
        "20"
        + str(int(predict_year[2:4]) - 5).zfill(2)
        + "_"
        + str(int(predict_year[5:7]) - 5).zfill(2)
    )
else:
    train_year = predict_year[:4]



GT_TIF = DATA_DIR / f"training_data_x_{train_year}_y_{predict_year}.tif"
SITES_CSV = DATA_DIR / "Odisha_sites.csv"

LABEL_BAND = "9_deforestation"
WINDOW_SIZE = 1024


with rasterio.open(GT_TIF) as gt_src:
   
    band_names = list(gt_src.descriptions)
    label_idx = band_names.index(LABEL_BAND)
    gt = gt_src.read(label_idx + 1).astype(np.float32)
    transform = gt_src.transform
    crs = gt_src.crs

np.random.seed(42)

def_idx = np.column_stack(np.where(gt == 1))
no_def_idx = np.column_stack(np.where(gt == 0))

n = min(len(def_idx), 5000)  # cap if needed

sampled_def = def_idx[np.random.choice(len(def_idx), n, replace=False)]
sampled_no_def = no_def_idx[np.random.choice(len(no_def_idx), n, replace=False)]

pixels = np.vstack([sampled_def, sampled_no_def])
labels = np.hstack([np.ones(n), np.zeros(n)]).astype(int)


# Convert pixel row and column indices to geographic x,y coordinates using the raster transform
def pixel_to_xy(rows, cols, transform):
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return np.array(xs), np.array(ys)


rows = pixels[:, 0]
cols = pixels[:, 1]

xs, ys = pixel_to_xy(rows, cols, transform)


df = pd.DataFrame({
    "row": rows,
    "col": cols,
    "x": xs,
    "y": ys,
    "ground_truth": labels
})


df["pixel_id"] = np.arange(len(df))
df["crs"] = str(crs)


block_size = 16  # pixels

df["block_row"] = df["row"] // block_size
df["block_col"] = df["col"] // block_size
df["block_id"] = (
    df["block_row"].astype(str) + "_" +
    df["block_col"].astype(str)
)


df.to_csv("sampled_ground_truth_pixels.csv", index=False)

