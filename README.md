# Hyperspectral Soil Property Prediction Pipeline

A complete end-to-end Python pipeline for predicting soil properties from hyperspectral imagery using spectral preprocessing, chemometric feature selection, and machine learning regression models.

\---

## Pipeline Overview

```
Raw Hyperspectral Spectra (N samples × B bands)
         │
         ▼
┌─────────────────────────────────────────────┐
│           PREPROCESSING                     │
│  ALS Baseline → UV Norm → SG2 → SNV         │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│       KENNARD-STONE SPLIT  (70 / 30)        │
│  Deterministic · Spectral-space based       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│          FEATURE SELECTION                  │
│  CARS  ·  SPA  ·  Simulated Annealing       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│          REGRESSION MODELS                  │
│  PLSR  ·  SVR  ·  RF  ·  XGBoost           │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│           OUTPUTS                           │
│  Metrics · Plots · CSVs · Excel             │
└─────────────────────────────────────────────┘
```

\---

## Methods

### Preprocessing

|Step|Method|Purpose|
|-|-|-|
|1|**ALS** — Asymmetric Least Squares|Baseline/scatter correction|
|2|**UV** — Unit Vector Normalization|Scale-invariant representation|
|3|**SG2** — Savitzky-Golay 2nd Derivative|Noise reduction, peak sharpening|
|4|**SNV** — Standard Normal Variate|Remove multiplicative scatter effects|

### Data Splitting

**Kennard-Stone algorithm** selects training samples that uniformly cover the spectral feature space — unlike random splits, it guarantees the training set spans the full spectral variation. 70% train / 30% test, fully deterministic.

### Feature Selection

|Method|Strategy|
|-|-|
|**CARS**|Competitive Adaptive Reweighted Sampling — Monte Carlo + Exponential Decay + ARS|
|**SPA**|Successive Projections Algorithm — projection-based collinearity minimization|
|**SA**|Simulated Annealing — global optimization across multiple subset sizes|

All three use 5-fold CV RMSECV on the training set only — zero test leakage.

### Regression Models

|Model|Description|
|-|-|
|**PLSR**|Partial Least Squares Regression — components auto-selected by RMSECV|
|**SVR**|Support Vector Regression (RBF kernel)|
|**RF**|Random Forest (150 trees, sqrt features)|
|**XGBoost**|Gradient Boosted Trees (depth=3, lr=0.03)|

### Evaluation Metrics

R² · RMSE · MAE · RPD · RPIQ · MBE computed for both train and test sets across all model–feature combinations.

\---

## Repository Structure

```
hyperspectral-soil-prediction/
│
├── pipeline.py                  ← Main script (edit paths at top)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── generate\\\_sample\\\_data.py  ← Creates synthetic demo CSV
│   └── README.md                ← Input format specification
│
└── outputs/                     ← Generated at runtime (not tracked)
    ├── PLOTS/                   ← 16 figures @ 600 dpi
    ├── CSV/                     ← Metrics, obs vs pred, split indices
    └── FEATURES/                ← Selected wavelength indices per method
```

\---

## Quickstart

### 1\. Clone the repository

```bash
git clone https://github.com/abhi1earth/hyperspectral-soil-prediction.git
cd hyperspectral-soil-prediction
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Generate synthetic demo data

```bash
python data/generate\\\_sample\\\_data.py
```

### 4\. Update the input path in `pipeline.py`

```python
# Line 30 in pipeline.py
INPUT\\\_FILE = "data/sample\\\_data.csv"
```

### 5\. Run the pipeline

```bash
python pipeline.py
```

All outputs are written to the `outputs/` directory.

\---

## Using Your Own Data

Prepare a CSV with this structure:

```
Sample\\\_ID, 400.0, 402.5, 405.0, ..., 2500.0, Target
S001,       0.213, 0.198, 0.245, ..., 0.189,  3.12
```

* **First column**: sample identifier (string)
* **Middle columns**: spectral bands, headers = wavelength in nm (numeric)
* **Last column**: soil property to predict (numeric)

Then update `INPUT\\\_FILE` in `pipeline.py` and run.

\---

## Outputs

After a successful run, the `outputs/` folder contains:

**Plots (PLOTS/)**

|File|Content|
|-|-|
|`01–05\\\_Spectrum\\\_\\\*.png`|Mean spectra at each preprocessing stage|
|`DIAG\\\_target\\\_distribution.png`|Train vs Test target histogram (Kennard-Stone)|
|`DIAG\\\_mean\\\_spectra.png`|Train vs Test mean spectra overlay|
|`DIAG\\\_band\\\_correlation.png`|Pearson r per band vs target property|
|`DIAG\\\_PLS\\\_RMSECV\\\_vs\\\_components.png`|Component selection curve|
|`FS\\\_CARS\\\_SelectedVariables.png`|CARS selected wavelengths on mean spectrum|
|`FS\\\_CARS\\\_RMSECV.png`|RMSECV vs MC run number|
|`FS\\\_CARS\\\_NumVariables\\\_RMSECV.png`|RMSECV vs number of selected variables|
|`FS\\\_SPA\\\_SelectedVariables.png`|SPA selected wavelengths|
|`FS\\\_SPA\\\_RMSECV.png`|SPA RMSECV curve|
|`FS\\\_SA\\\_SelectedVariables.png`|SA selected wavelengths|
|`FS\\\_SA\\\_RMSECV.png`|SA RMSECV vs subset size|

**Tables (CSV/)**

* `FINAL\\\_Model\\\_Comparison\\\_TRAIN.csv` — all train metrics
* `FINAL\\\_Model\\\_Comparison\\\_TEST.csv` — all test metrics
* `ObsPred\\\_{tag}\\\_{model}.csv` — observed vs predicted per model (12 files)
* `Split\\\_Indices.csv` — Kennard-Stone sample assignment
* `DIAG\\\_PLS\\\_RMSECV\\\_vs\\\_components.csv` — component selection data

**Excel**

* `Pipeline\\\_Comparison.xlsx` — 3 sheets: Raw+NoFS | Prep+NoFS | Prep+FS

\---

## Comparison Matrix (example with synthetic data)

Results vary by dataset. The pipeline is designed to be compared across three scenarios:

|Scenario|Description|
|-|-|
|**S1**|Raw spectra · No feature selection|
|**S2**|ALS+UV+SG2+SNV · No feature selection|
|**S3**|ALS+UV+SG2+SNV + CARS / SPA / SA|

\---

## Citation

This pipeline supports research currently under peer review. Citation details will be added upon acceptance.

If you use this code in your own work, please cite this repository:

```
Chakraborty, A. (2026). Hyperspectral Soil Property Prediction Pipeline.
GitHub. https://github.com/abhi1earth/hyperspectral-soil-prediction
```

\---

## Author

**Abhishek Chakraborty**
MSc Geography (Remote Sensing \& GIS) · CRAN package author ([VegSpecIndex](https://cran.r-project.org/package=VegSpecIndex))
GitHub: [@abhi](https://github.com/abhisek-santra)1earth

\---

## License

MIT License — see [LICENSE](LICENSE) for details.

