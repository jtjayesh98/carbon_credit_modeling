# Dhenkanal Counterfactual Deforestation Modeling

This project implements a comprehensive workflow for estimating deforestation and measuring additionality in the Dhenkanal region of Odisha, India. It combines satellite imagery analysis with machine learning and counterfactual estimation techniques.

## Workflow Overview

### 1. Data Extraction from Google Earth Engine (GEE)

**Status**: [PLACEHOLDER - Add data extraction details here]

- Source: Google Earth Engine
- Data extracted for Dhenkanal region
- Output directory: `data/GEE_exports_Dhenkanal/`
- Key datasets:
  - Forest cover rasters (2005, 2010, 2015)
  - Training data with labeled pixels
  - District and jurisdiction boundaries

### 2. Model Training

**Script**: `model_tif.py`

Train Random Forest models on labeled training data:
- Input: Training data from GEE with labeled deforestation pixels
- Process:
  - Load training data with multiple bands (spectral indices, forest cover, etc.)
  - Split data into features and labels
  - Train RF models for different time periods (ex-ante and ex-post)
  - Apply SMOTE for class imbalance handling
  - Save trained models to `models/` directory
- Output: 
  - `rf_smote_x_2000_05_y_2005_10_ex_ante.joblib`
  - `rf_smote_x_2010_y_2010_15_ex_post.joblib`

### 3. Counterfactual Estimation

**Script**: `counterfactual.py`

Generate counterfactual scenarios to estimate what deforestation would have occurred without intervention:
- Input: Trained RF models and raster data
- Process:
  - Create counterfactual predictions using ex-ante models on ex-post time periods
  - Compare actual deforestation (from models) vs. counterfactual scenarios
  - Estimate additionality (avoided deforestation)
- Output: Counterfactual prediction rasters saved to `outputs/predictions/`

### 4. Prediction Generation

**Script**: `pred_raster.py`

Generate deforestation probability maps for the modeling region:
- Input: Trained RF models and full raster data
- Process:
  - Apply models to full raster extents (or specific sites)
  - Generate probability maps for different scenarios (ex-ante, ex-post)
  - Optionally adjust predictions to match observed deforestation quantities
  - Support both site-only and full-raster predictions
- Output: 
  - Deforestation probability rasters: `outputs/predictions/deforestation_prob_*.tif`
  - Adjusted/density maps
  - Class predictions (binary deforestation)

### 5. Evaluation, Visualization & Analysis

#### Area Calculations

**Script**: `calculate_forested_area.py`

Calculate forested and deforested areas for each Odisha site:
- Input: 
  - `Dhenkanal_2010.tif` (forest extent)
  - `training_data_x_2010_y_2010_15.tif` (ground truth deforestation)
  - `Odisha_sites.csv` (site geometries)
- Process:
  - Total area: All pixels from Dhenkanal_2010 within site geometry
  - Forested area: Pixels with value 1 from Dhenkanal_2010
  - Deforested area: Pixels with value 1 from "9_deforestation" band
- Output: `forested_and_deforested_area_by_region.csv`

#### Visualization Scripts

**Ground Truth Deforestation**

**Script**: `visualize_gt_deforestation.py`

Visualize the ground truth deforestation patterns:
- Input: Ground truth raster with "9_deforestation" band and site geometries
- Process:
  - Reproject to high-resolution grid (0.0001 degrees)
  - Mask to individual site boundaries
  - Count deforested pixels and calculate area
  - Generate binary deforestation maps
- Output: 
  - PDF and JPG visualizations: `figures/gt_deforestation_*.{pdf,jpg}`
  - Console output of deforested pixel counts per site

**Forest Cover Visualization**

**Script**: `visualize_forest_cover.py`

Visualize forest extent from Dhenkanal rasters for different time periods.

**Model Prediction Visualization**

**Script**: `visualize_odisha_sites.py`

Visualize RF model predictions and counterfactual scenarios:
- Generates prediction maps for all sites
- Supports different model types (RF model, counterfactual)
- Output: `figures/model_prediction_*.{pdf,jpg}`, `figures/cf_prediction_*.{pdf,jpg}`

#### Additionality Estimation

**Script**: `additionality.py`

Calculate additionality metrics (avoided deforestation):
- Input: Counterfactual scenarios and actual predictions
- Process:
  - Compare counterfactual vs. actual deforestation
  - Calculate additionality for each site
  - Generate summary statistics
- Output: `additionality_results.json`, `additionality_table.csv`

#### Additional Analysis

**Scripts**:
- `area_estimation.py`: Estimate deforested areas for different model runs
- `evaluate_predictions.py`: Evaluate model performance metrics
- `run_all_evaluations.py`: Run comprehensive evaluation suite
- `artificial_sites.py`: Analysis for artificial/synthetic sites
- `file_analysis.py`: Data quality checks and file analysis

## Directory Structure

```
CCModeling/
├── data/
│   ├── images/               # Raster data and training data
│   ├── GEE_exports_Dhenkanal/  # Raw exports from GEE
│   └── graph/                # Intermediate data
├── models/                   # Trained RF models (.joblib)
├── outputs/
│   └── predictions/          # Generated prediction rasters
├── figures/                  # Visualization outputs (PDF, JPG)
├── modeling/                 # Python virtual environment
├── archives/                 # Legacy/archived code
└── [Python scripts]          # Main processing scripts
```

## Key Files

- **Input Data**: `data/images/Dhenkanal_*.tif`, `training_data_*.tif`, `Odisha_sites.csv`
- **Models**: `models/rf_smote_*.joblib`
- **Results**: `additionality_table.csv`, `forested_and_deforested_area_by_region.csv`
- **Visualizations**: `figures/gt_deforestation_*.{pdf,jpg}`, `figures/model_prediction_*.{pdf,jpg}`

## Usage

1. **Extract data from GEE** (see placeholder section above)
2. **Train models**: `python model_tif.py`
3. **Generate counterfactuals**: `python counterfactual.py`
4. **Generate predictions**: `python pred_raster.py`
5. **Calculate areas**: `python calculate_forested_area.py`
6. **Visualize results**: `python visualize_gt_deforestation.py`, `python visualize_odisha_sites.py`
7. **Run full evaluation**: `python run_all_evaluations.py`

## Key Metrics

- **Total Area**: All pixels in Dhenkanal_2010 within site geometry
- **Forested Area**: Pixels with value 1 (forest)
- **Deforested Area**: Ground truth deforestation from "9_deforestation" band
- **Pixel Size**: 0.09 hectares per pixel
- **Additionality**: Estimated avoided deforestation from counterfactual analysis

## Configuration

- **Pixel Area**: 0.09 ha (based on ~30m resolution native data)
- **Coordinate System**: EPSG:4326 (WGS84)
- **High-resolution grid for visualization**: 0.0001 degrees (~10m)
- **Model Parameters**: RF with SMOTE for class balancing

## Requirements

See `requirements.txt` for Python dependencies (rasterio, geopandas, scikit-learn, matplotlib, etc.)

## Notes

- All areas are calculated in hectares
- Ground truth data spans 2010-2015 period
- Models trained for both ex-ante (baseline) and ex-post (intervention) scenarios
- Counterfactual analysis estimates what would have happened without intervention
