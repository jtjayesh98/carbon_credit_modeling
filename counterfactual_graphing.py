import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


GRAPH_DIR = Path("./data/graph/")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    '0_Rainfall_norm', '1_Mean_Rainfall', '2_NDVI', '3_EVI',
    '4_ground_temp', '5_edge_distance', '6_elevation',
    '7_slope', '8_deforestation_density'
]

OUTCOME = "9_deforestation"
PS_COL = "propensity"


# Plot propensity score overlap between treated and control groups
def plot_ps_overlap(treated, control):
    plt.figure(figsize=(6, 4))
    plt.hist(control[PS_COL], bins=50, alpha=0.6, label="Control", density=True)
    plt.hist(treated[PS_COL], bins=50, alpha=0.6, label="Treated", density=True)
    plt.xlabel("Propensity score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "propensity_score_overlap.png", dpi=300)
    plt.close()


# Plot the distribution of counterfactual outcomes for treated units
def plot_counterfactual_outcomes(treated):
    plt.figure(figsize=(6, 4))
    plt.hist(treated["y_cf"], bins=40, alpha=0.7, label="Counterfactual outcome")
    plt.axvline(treated[OUTCOME].mean(), linestyle="--", label="Mean treated outcome")
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "counterfactual_outcome_distribution.png", dpi=300)
    plt.close()


# Plot counterfactual feature distributions for each feature
def plot_feature_distributions(treated, control, indices):
    for j, feature in enumerate(FEATURES):
        cf_vals = []
        for idxs in indices:
            cf_vals.extend(control.iloc[idxs][feature].values)

        plt.figure(figsize=(5, 4))
        plt.boxplot(cf_vals, vert=False)
        plt.axvline(treated[feature].mean(), linestyle="--", label="Mean treated")
        plt.xlabel(feature)
        plt.legend()
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / f"feature_cf_{feature}.png", dpi=300)
        plt.close()



# Plot standardized differences for features between treated and counterfactual control
def plot_standardized_differences(treated, control, indices):
    std_diffs = []

    for feature in FEATURES:
        cf_vals = []
        for idxs in indices:
            cf_vals.extend(control.iloc[idxs][feature].values)

        cf_vals = np.array(cf_vals)
        diff = (treated[feature].mean() - cf_vals.mean()) / cf_vals.std()
        std_diffs.append(diff)

    plt.figure(figsize=(7, 4))
    plt.axhline(0, color="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Standardized difference")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "standardized_feature_differences.png", dpi=300)
    plt.close()



# Run all graphing functions for counterfactual analysis
def run_all_graphs(treated, control, indices):
    print(indices)
    plot_ps_overlap(treated, control)
    plot_counterfactual_outcomes(treated)
    plot_feature_distributions(treated, control, indices)
    plot_standardized_differences(treated, control, indices)
