import rasterio
import rasterio.windows
from rasterio.features import geometry_mask

import numpy as np
import pandas as pd
import json

from shapely.geometry import shape

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import shap
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

import matplotlib.pyplot as plt
import joblib
from pathlib import Path

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


predict_year = "2010_15"
ex_ante = False

if ex_ante:
    train_year = "20" + str(int(predict_year[2:4]) - 5).zfill(2) + "_" + \
                 str(int(predict_year[5:7]) - 5).zfill(2)
else:
    train_year = predict_year[:4]

print(train_year, "->", predict_year)

data_preprocessing = "smote"  # regular | undersampling | oversampling | smote

raster_path = f"./data/images/training_data_x_{train_year}_y_{predict_year}.tif"

print(raster_path)

sites_csv_path = Path(raster_path).parent / "Odisha_sites.csv"

FEATURE_BANDS = [
    "0_Rainfall_norm", "1_Mean_Rainfall", "2_NDVI", "3_EVI",
    "4_ground_temp", "5_edge_distance", "6_elevation",
    "7_slope", "8_deforestation_density"
]

LABEL_BAND = "9_deforestation"
MAX_SAMPLES = 300_000


sites_df = pd.read_csv(sites_csv_path)

site_geoms = []
for g in sites_df[".geo"]:
    site_geoms.append(shape(json.loads(g)))

print(f"Loaded {len(site_geoms)} Odisha site polygons")


# Generate windows for processing large rasters in chunks
def window_generator(src, window_size=1024):
    for row in range(0, src.height, window_size):
        for col in range(0, src.width, window_size):
            yield rasterio.windows.Window(
                col, row,
                min(window_size, src.width - col),
                min(window_size, src.height - row)
            )


X_list, y_list = [], []

with rasterio.open(raster_path) as src:

    band_names = list(src.descriptions)
    feature_idxs = [band_names.index(b) for b in FEATURE_BANDS]
    label_idx = band_names.index(LABEL_BAND)

    for window in window_generator(src):

        chunk = src.read(window=window)
        h, w = chunk.shape[1], chunk.shape[2]

        site_mask = geometry_mask(
            site_geoms,
            transform=src.window_transform(window),
            invert=True,            # True = inside polygon
            out_shape=(h, w)
        )

        pixels = chunk.reshape(chunk.shape[0], -1).T
        X = pixels[:, feature_idxs]
        y = pixels[:, label_idx]

        site_mask_flat = site_mask.flatten()

        valid_mask = (
            (~np.isnan(X).any(axis=1)) &
            (~np.isnan(y)) &
            (~site_mask_flat)
        )

        X, y = X[valid_mask], y[valid_mask]

        if len(X) > MAX_SAMPLES:
            idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            X, y = X[idx], y[idx]

        X_list.append(X)
        y_list.append(y)

X = np.vstack(X_list)
y = np.concatenate(y_list)

print("Final dataset shape:", X.shape)
print("Label distribution:", np.unique(y, return_counts=True))


if data_preprocessing == "undersampling":
    X, y = RandomUnderSampler(random_state=42).fit_resample(X, y)

elif data_preprocessing == "oversampling":
    X, y = RandomOverSampler(random_state=42).fit_resample(X, y)

elif data_preprocessing == "smote":
    X, y = SMOTE(random_state=42).fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(
    n_estimators=20,
    max_depth=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")

pred = model.predict(X_test)
pred_prob = model.predict_proba(X_test)

print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, pred_prob[:, 1]))
print("Accuracy:", accuracy_score(y_test, pred))
print("CV mean:", scores.mean(), "std:", scores.std())


background = shap.utils.sample(X_train[:20000], 200, random_state=0)

explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_names=FEATURE_BANDS,
    model_output="probability"
)

sv = explainer(X_train[:20000])

shap.plots.beeswarm(sv[..., 1], max_display=20)
plt.savefig(f"beeswarm_{predict_year}_{'ex_ante' if ex_ante else 'ex_post'}.png",
            dpi=300, bbox_inches="tight")
plt.close()

shap.plots.bar(sv[..., 1], max_display=20)
plt.savefig(f"bar_{predict_year}_{'ex_ante' if ex_ante else 'ex_post'}.png",
            dpi=300, bbox_inches="tight")
plt.close()


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

suffix = "ex_ante" if ex_ante else "ex_post"
model_path = MODEL_DIR / f"rf_{data_preprocessing}_x_{train_year}_y_{predict_year}_{suffix}.joblib"

joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
