import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from utilities import tif_to_dataframe_with_treatment, dataframe_to_multiband_tif
from counterfactual_graphing import run_all_graphs
from pathlib import Path
from datetime import datetime

# Log progress messages with timestamps
def log_progress(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Shift a year string back by specified 5-year periods
def shift_year_back(year_str, periods=1):
    """Shift a 'YYYY_YY' string back by `periods` 5-year periods."""
    start = int("20" + year_str[2:4]) - 5 * periods
    end   = int("20" + year_str[5:7]) - 5 * periods
    return f"{start}_{str(end)[2:]}"

predict_year = "2010_15"
ex_ante = True

log_progress("Starting counterfactual analysis pipeline")
log_progress(f"Predict year: {predict_year}, Ex-ante: {ex_ante}")

POLYGON_PATH = "./data/images/Odisha_sites.csv"
covariate_columns = [
    '0_Rainfall_norm', '1_Mean_Rainfall', '2_NDVI', '3_EVI',
    '4_ground_temp', '5_edge_distance', '6_elevation', '7_slope',
    '8_deforestation_density'
]
label = "9_deforestation"
suffix = "ante" if ex_ante else "post"

if ex_ante:
    ps_x_year    = shift_year_back(predict_year, periods=2)   # 2000_05
    ps_y_year    = shift_year_back(predict_year, periods=1)   # 2005_10
    pred_x_year  = shift_year_back(predict_year, periods=1)   # 2005_10
    pred_y_year  = predict_year                               # 2010_15

    PS_TIF   = f"./data/images/training_data_x_{ps_x_year}_y_{ps_y_year}.tif"
    DATA_TIF = f"./data/images/training_data_x_{pred_x_year}_y_{pred_y_year}.tif"
    log_progress(f"Ex-ante mode | PS TIF: {PS_TIF}")
    log_progress(f"Ex-ante mode | Predict TIF: {DATA_TIF}")
else:
    train_year = predict_year[:4]
    DATA_TIF   = f"./data/images/training_data_x_{train_year}_y_{predict_year}.tif"
    PS_TIF     = DATA_TIF          # same file — original behaviour
    log_progress(f"Post mode | TIF: {DATA_TIF}")

log_progress("Loading propensity-score TIF...")
df_ps = tif_to_dataframe_with_treatment(tif_path=PS_TIF, polygon_csv_path=POLYGON_PATH, all_touched=True)
df_ps = df_ps.dropna().reset_index(drop=True)
log_progress(f"PS data loaded: {len(df_ps)} pixels")

X_ps = df_ps[covariate_columns].values
T_ps = df_ps['Treatment'].values

ps_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])
log_progress("Fitting propensity score model...")
ps_model.fit(X_ps, T_ps)
log_progress("Propensity score model fitted")




if ex_ante:
    log_progress("Loading prediction TIF (shifted period)...")
    df = tif_to_dataframe_with_treatment(tif_path=DATA_TIF, polygon_csv_path=POLYGON_PATH, all_touched=True)
    df = df.dropna().reset_index(drop=True)
    log_progress(f"Prediction data loaded: {len(df)} pixels")

    log_progress("Transferring propensity scores via geometric nearest-neighbour...")

    ps_coords   = df_ps[['row', 'col']].values       # pixel grid coordinates
    pred_coords = df[['row', 'col']].values          # pixel grid coordinates        # (N_pred, 2)

    geo_nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    geo_nn.fit(ps_coords)
    geo_dist, geo_idx = geo_nn.kneighbors(pred_coords)

    ps_scores_all = ps_model.predict_proba(df_ps[covariate_columns].values)[:, 1]
    df["propensity"] = ps_scores_all[geo_idx.flatten()]

    log_progress(f"Propensity scores transferred (max geo dist: {geo_dist.max():.4f})")
else:
    df = df_ps.copy()          # same dataframe, original behaviour
    df["propensity"] = ps_model.predict_proba(df[covariate_columns].values)[:, 1]

df = df.reset_index(drop=True)
df["flat_index"] = df.index

treated = df[df['Treatment'] == 1].copy()
control = df[df['Treatment'] == 0].copy()
log_progress(f"Initial counts — Treated: {len(treated)}, Control: {len(control)}")

treated = treated[
    (treated['propensity'] > control['propensity'].min()) &
    (treated['propensity'] < control['propensity'].max())
]
log_progress(f"After common support trimming — Treated: {len(treated)}, Control: {len(control)}")
treated = treated.reset_index(drop=True)
control = control.reset_index(drop=True)

k = 20
log_progress(f"Fitting NN model for treated group (k={k})...")
nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
nn.fit(control[['propensity']])
distances, indices = nn.kneighbors(treated[['propensity']])

control_y_values  = control[label].values
treated_y_values  = treated[label].values

y_cf_treated  = np.array([control_y_values[idxs].mean() for idxs in indices])
delta_y_treated = y_cf_treated - treated_y_values
ATT = np.mean(delta_y_treated)

treated["y_cf"]            = y_cf_treated
treated["delta_y"]         = delta_y_treated
treated["match_dist_mean"] = distances.mean(axis=1)
treated["match_dist_max"]  = distances.max(axis=1)
treated["n_matches"]       = k

log_progress(f"ATT: {ATT:.6f}")

log_progress("Fitting NN model for control group...")
nn_control = NearestNeighbors(n_neighbors=min(k, len(treated)), metric='euclidean')
nn_control.fit(treated[['propensity']])
distances_control, indices_control = nn_control.kneighbors(control[['propensity']])

treated_y_for_control = treated[label].values
y_cf_control   = np.array([treated_y_for_control[idxs].mean() for idxs in indices_control])
delta_y_control = y_cf_control - control[label].values

control["y_cf"]    = y_cf_control
control["delta_y"] = delta_y_control

df["y_cf"]    = np.nan
df["delta_y"] = np.nan
df.loc[treated.index, ["y_cf", "delta_y"]] = treated[["y_cf", "delta_y"]].values
df.loc[control.index, ["y_cf", "delta_y"]] = control[["y_cf", "delta_y"]].values

log_progress("Writing output TIF...")
with rasterio.open(DATA_TIF) as src:
    height, width = src.height, src.width
    transform, crs = src.transform, src.crs

output_file = f"./outputs/predictions/counterfactual_prediction_FULL_{predict_year}_ex_{suffix}.tif"
dataframe_to_multiband_tif(
    df=df,
    value_cols=["9_deforestation", "y_cf", "delta_y", "Treatment"],
    output_tif=output_file,
    height=height, width=width,
    transform=transform, crs=crs,
)
log_progress(f"Output TIF saved: {output_file}")

log_progress("Generating analysis graphs...")
run_all_graphs(treated, control, indices)
log_progress("Pipeline completed!")