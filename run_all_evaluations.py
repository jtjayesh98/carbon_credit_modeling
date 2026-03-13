import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import json

from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, Point
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "images"
PRED_DIR = BASE_DIR / "outputs" / "predictions"

SAMPLED_PIXELS_CSV = DATA_DIR / "sampled_ground_truth_pixels.csv"
LABEL_BAND = "9_deforestation"

@dataclass
class RunConfig:
    name: str
    ex_ante: bool
    counterfactual: bool
    udef_arp: bool


RUNS = [
    RunConfig("ex_ante", True, False, False),
    RunConfig("ex_post", False, False, False),
    RunConfig("counterfactual_ex_ante", True, True, False),
    RunConfig("counterfactual_ex_post", False, True, False),
    RunConfig("udef_arp", False, False, True),
]

COMMON_MASK = None

def build_common_mask():
    predict_year = "2010_15"
    masks = []

    for cfg in RUNS:

        if cfg.ex_ante:
            train_year = (
                "20"
                + str(int(predict_year[2:4]) - 5).zfill(2)
                + "_"
                + str(int(predict_year[5:7]) - 5).zfill(2)
            )
        else:
            train_year = predict_year[:4]

        gt_tif = DATA_DIR / f"training_data_x_{train_year}_y_{predict_year}.tif"

        with rasterio.open(gt_tif) as src:
            label_idx = list(src.descriptions).index(LABEL_BAND)
            gt = src.read(label_idx + 1)
            masks.append(~np.isnan(gt))

    global COMMON_MASK
    COMMON_MASK = np.logical_and.reduce(masks)


build_common_mask()


def evaluate_run(cfg: RunConfig) -> dict:

    predict_year = "2010_15"

    if cfg.ex_ante:
        train_year = (
            "20"
            + str(int(predict_year[2:4]) - 5).zfill(2)
            + "_"
            + str(int(predict_year[5:7]) - 5).zfill(2)
        )
        suffix = "ex_ante"
    else:
        train_year = predict_year[:4]
        suffix = "ex_post"

    if cfg.udef_arp:
        pred_tif = PRED_DIR / "Acre_Adjucted_Density_Map_VP.tif"
    elif cfg.counterfactual:
        pred_tif = PRED_DIR / f"counterfactual_prediction_FULL_{predict_year}_ex_{'ante' if cfg.ex_ante else 'post'}.tif"
    else:
        pred_tif = PRED_DIR / f"deforestation_class_{predict_year}_full_{suffix}.tif"

    gt_tif = DATA_DIR / f"training_data_x_{train_year}_y_{predict_year}.tif"

    samples = pd.read_csv(SAMPLED_PIXELS_CSV)

    if not cfg.ex_ante:
        sites_df = pd.read_csv(DATA_DIR / "Odisha_sites.csv")
        site_geoms = [shape(json.loads(g)) for g in sites_df[".geo"]]
        
        mask = []
        for _, row in samples.iterrows():
            point = Point(row['x'], row['y'])
            inside = any(geom.contains(point) for geom in site_geoms)
            mask.append(not inside)
        
        samples = samples[mask].reset_index(drop=True)

    rows = samples["row"].values.astype(int)
    cols = samples["col"].values.astype(int)
    y_true_all = samples["ground_truth"].values.astype(int)

    with rasterio.open(gt_tif) as gt_src:
        label_idx = list(gt_src.descriptions).index(LABEL_BAND)

    y_pred = []
    y_true = []

    with rasterio.open(gt_tif) as gt_src, rasterio.open(pred_tif) as pred_src:

        if cfg.udef_arp:
            aligned_pred = np.empty(
                (gt_src.height, gt_src.width),
                dtype=np.float32
            )

            reproject(
                source=rasterio.band(pred_src, 1),
                destination=aligned_pred,
                src_transform=pred_src.transform,
                src_crs=pred_src.crs,
                dst_transform=gt_src.transform,
                dst_crs=gt_src.crs,
                resampling=Resampling.nearest
            )

            pixel_area_ha = (30 * 30) / 10000.0
            aligned_pred[aligned_pred < 0] = 0
            rel_prob = aligned_pred / pixel_area_ha
            rel_prob = rel_prob / np.nanmax(rel_prob)

            pred_bin = (rel_prob >= 0.5).astype(np.uint8)

        else:
            if cfg.counterfactual:
                band_names = list(pred_src.descriptions)
                if "y_cf" in band_names:
                    band_idx = band_names.index("y_cf") + 1
                else:
                    band_idx = 2

                pred_vals = pred_src.read(band_idx)
                valid_mask = ~np.isnan(pred_vals)
                thresh = np.percentile(pred_vals[valid_mask], 50)
                pred_bin = (pred_vals >= thresh).astype(np.uint8)

            else:
                pred_bin = pred_src.read(1).astype(np.uint8)

        gt_full = gt_src.read(label_idx + 1)

        if not cfg.udef_arp:
            sample_mask = ~np.isnan(gt_full)
            agreement = np.mean(pred_bin[sample_mask] == gt_full[sample_mask])
            print(f"[DEBUG] Agreement: {agreement:.4f}")
            if agreement > 0.98:
                print("⚠️ WARNING: Possible leakage detected")

        for i, (r, c) in enumerate(zip(rows, cols)):

            if not COMMON_MASK[r, c]:
                continue

            y_pred.append(pred_bin[r, c])
            y_true.append(y_true_all[i])

    y_pred = np.array(y_pred).astype(int)
    y_true = np.array(y_true).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "run": cfg.name,
        "ex_ante": cfg.ex_ante,
        "counterfactual": cfg.counterfactual,
        "udef_arp": cfg.udef_arp,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n_eval": len(y_true),
    }


results = []

for cfg in RUNS:
    print(f"Running: {cfg.name}")
    res = evaluate_run(cfg)
    results.append(res)

df = pd.DataFrame(results)

out_csv = BASE_DIR / "comparative_evaluation_metrics.csv"
out_parquet = BASE_DIR / "comparative_evaluation_metrics.parquet"

df.to_csv(out_csv, index=False)
df.to_parquet(out_parquet)

print("\n=== Comparative Evaluation Results ===")
print(df)