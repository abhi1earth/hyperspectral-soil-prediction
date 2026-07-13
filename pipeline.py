# ==============================================================
# Hyperspectral Soil Property Prediction Pipeline
# ==============================================================
# Preprocessing : ALS > UV Normalization > SG2 > SNV
# Data Split    : Kennard-Stone 70/30 (deterministic, spectral-space)
# Feature Sel.  : CARS (MC+EDF+ARS) | SPA (projection) | SA (annealing)
# Models        : PLSR | SVR | RF | XGB
# Metrics       : R², RMSE, MAE, RPD, RPIQ, MBE
# Output        : Plots (600 dpi), CSVs, 3-sheet Excel
# ==============================================================
# Author : Abhishek Chakraborty
# GitHub : https://github.com/abhi1earth
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (mean_squared_error, r2_score,
                              mean_absolute_error, pairwise_distances)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

np.random.seed(42)

# ==============================================================
# CONFIGURATION — edit these paths before running
# ==============================================================
INPUT_FILE  = "data/input_spectra.csv"   # columns: ID | band_1 ... band_N | target
BASE_DIR    = "outputs"                  # all results written here
TARGET_COL  = -1                         # last column = soil property to predict

CSV_DIR   = os.path.join(BASE_DIR, "CSV")
PLOT_DIR  = os.path.join(BASE_DIR, "PLOTS")
FEAT_DIR  = os.path.join(BASE_DIR, "FEATURES")
for d in [CSV_DIR, PLOT_DIR, FEAT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================================================
# PLOT STYLE
# ==============================================================
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         False,
    "savefig.facecolor": "white",
    "font.size":         11,
})

def clean_ax(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

# ==============================================================
# LOAD DATA
# ==============================================================
df          = pd.read_csv(INPUT_FILE)
IDs         = df.iloc[:, 0].values
y           = df.iloc[:, TARGET_COL].values.astype(float)
wavenumbers = df.columns[1:TARGET_COL].astype(float).values
X_raw       = df.iloc[:, 1:TARGET_COL].values.astype(float)

print(f"Loaded : {X_raw.shape[0]} samples × {X_raw.shape[1]} bands")
print(f"Target — mean:{y.mean():.3f}  std:{y.std():.3f}  "
      f"min:{y.min():.3f}  max:{y.max():.3f}")

# ==============================================================
# STEP 1 — ALS BASELINE CORRECTION
# ==============================================================
def als_baseline(spec, lam=1e5, p=0.01, niter=10):
    """Asymmetric Least Squares baseline estimation."""
    L = len(spec)
    D = np.zeros((L - 2, L))
    for i in range(L - 2):
        D[i, i:i+3] = [1, -2, 1]
    w = np.ones(L)
    for _ in range(niter):
        Z = np.linalg.solve(np.diag(w) + lam * (D.T @ D), w * spec)
        w = p * (spec > Z) + (1 - p) * (spec < Z)
    return Z

print("\nALS baseline correction...")
X_als = np.array([x - als_baseline(x) for x in tqdm(X_raw)])

# ==============================================================
# STEP 2 — UV + SG2 + SNV PREPROCESSING
# ==============================================================
# UV  : unit-vector normalization
X_uv  = X_als / np.linalg.norm(X_als, axis=1, keepdims=True)
# SG2 : Savitzky-Golay 2nd derivative (window=21, poly=2)
X_sg  = savgol_filter(X_uv, window_length=21, polyorder=2, deriv=2, axis=1)
# SNV : standard normal variate
X_snv = (X_sg - X_sg.mean(axis=1, keepdims=True)) / X_sg.std(axis=1, keepdims=True)

pd.DataFrame(X_snv, columns=wavenumbers).to_csv(
    os.path.join(CSV_DIR, "Preprocessed_Spectra.csv"), index=False)
print("Preprocessing complete (ALS > UV > SG2 > SNV).")

# ==============================================================
# PREPROCESSING PLOTS
# ==============================================================
def save_preproc_plot(X, label, filename):
    ms = X.mean(axis=0)
    ss = X.std(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavenumbers, ms, color="#1f77b4", linewidth=1.2, label="Mean")
    ax.fill_between(wavenumbers, ms - ss, ms + ss,
                    alpha=0.15, color="#1f77b4", label="±1 SD")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Mean Spectrum – {label}")
    ax.legend(fontsize=9)
    clean_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

print("\nSaving preprocessing plots...")
save_preproc_plot(X_raw, "01_RAW", "01_Spectrum_RAW.png")
save_preproc_plot(X_als, "02_ALS", "02_Spectrum_ALS.png")
save_preproc_plot(X_uv,  "03_UV",  "03_Spectrum_UV.png")
save_preproc_plot(X_sg,  "04_SG2", "04_Spectrum_SG2.png")
save_preproc_plot(X_snv, "05_SNV", "05_Spectrum_SNV.png")

# ==============================================================
# STEP 3 — KENNARD-STONE 70/30 SPLIT
# ==============================================================
def kennard_stone(X, n_train):
    """
    Kennard-Stone algorithm: select n_train samples that uniformly
    cover the spectral feature space. Deterministic — no random seed needed.
    Returns (train_indices, test_indices).
    """
    dist = pairwise_distances(X)
    n    = X.shape[0]
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    sel  = [int(i), int(j)]
    rem  = [k for k in range(n) if k not in sel]
    while len(sel) < n_train:
        sub      = dist[np.ix_(rem, sel)]
        min_dist = sub.min(axis=1)
        pick     = rem[int(np.argmax(min_dist))]
        sel.append(pick)
        rem.remove(pick)
    return np.array(sel), np.array(rem)

print("\nKennard-Stone 70/30 split...")
n_train_ks          = int(0.70 * len(y))
train_idx, test_idx = kennard_stone(X_snv, n_train_ks)

X_train     = X_snv[train_idx];   X_test     = X_snv[test_idx]
X_train_raw = X_raw[train_idx];   X_test_raw = X_raw[test_idx]
y_train     = y[train_idx];       y_test     = y[test_idx]
ID_train    = IDs[train_idx];     ID_test    = IDs[test_idx]
print(f"  Train: {len(train_idx)}  |  Test: {len(test_idx)}")

# ==============================================================
# CROSS-VALIDATION OBJECT (shared across all CV steps)
# ==============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ==============================================================
# DIAGNOSTICS
# ==============================================================
print("\n=== DIAGNOSTICS ===")
print(f"y_train  mean:{y_train.mean():.3f}  std:{y_train.std():.3f}  "
      f"min:{y_train.min():.3f}  max:{y_train.max():.3f}")
print(f"y_test   mean:{y_test.mean():.3f}  std:{y_test.std():.3f}  "
      f"min:{y_test.min():.3f}  max:{y_test.max():.3f}")
print(f"NaN in X_train : {np.isnan(X_train).sum()}")
print(f"Inf in X_train : {np.isinf(X_train).sum()}")
out_of_range = np.sum((y_test < y_train.min()) | (y_test > y_train.max()))
print(f"Test samples outside train range: {out_of_range}/{len(y_test)}")

# ---- PLS component selection via RMSECV (training only, no leakage) ----
nc_range      = list(range(2, min(15, len(y_train) - 1)))
train_r2_list = []
rmsecv_list   = []

for nc in nc_range:
    pls    = PLSRegression(n_components=nc)
    ycv    = cross_val_predict(pls, X_train, y_train, cv=kf)
    rmsecv = np.sqrt(mean_squared_error(y_train, ycv))
    pls.fit(X_train, y_train)
    train_r2 = r2_score(y_train, pls.predict(X_train).ravel())
    train_r2_list.append(train_r2)
    rmsecv_list.append(rmsecv)

best_pos = int(np.argmin(rmsecv_list))
best_nc  = nc_range[best_pos]
best_nc  = max(2, min(best_nc, 10))

print(f"\nPLS RMSECV by component:")
for nc, tr, rv in zip(nc_range, train_r2_list, rmsecv_list):
    print(f"  nc={nc:2d} | train R²={tr:.4f} | RMSECV={rv:.4f}")
print(f"\nAuto-selected PLSR n_components = {best_nc}")

# ---- Band-wise correlation ----
band_corr = np.array([np.corrcoef(X_snv[:, i], y)[0, 1]
                       for i in range(X_snv.shape[1])])
print(f"Max |band-target correlation| : {np.abs(band_corr).max():.4f} "
      f"at {wavenumbers[np.abs(band_corr).argmax()]:.1f} nm")
print(f"Bands with |r| > 0.30 : {(np.abs(band_corr) > 0.30).sum()}")
print("===================\n")

# Diagnostic plots
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(y_train, bins=15, alpha=0.6, color="#1f77b4",
        label=f"Train n={len(y_train)}")
ax.hist(y_test,  bins=15, alpha=0.6, color="#ff7f0e",
        label=f"Test n={len(y_test)}")
ax.axvline(y_train.mean(), color="#1f77b4", linestyle="--", linewidth=1.2)
ax.axvline(y_test.mean(),  color="#ff7f0e", linestyle="--", linewidth=1.2)
ax.set_xlabel("Target Variable")
ax.set_ylabel("Count")
ax.set_title("Target Distribution — Train vs Test (Kennard-Stone)")
ax.legend(fontsize=9); clean_ax(ax); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "DIAG_target_distribution.png"),
            dpi=600, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(wavenumbers, X_train.mean(axis=0), color="#1f77b4",
        linewidth=1.2, label=f"Train n={len(y_train)}")
ax.plot(wavenumbers, X_test.mean(axis=0), color="#ff7f0e",
        linewidth=1.2, label=f"Test n={len(y_test)}")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity (SNV)")
ax.set_title("Mean Spectra — Train vs Test")
ax.legend(fontsize=9); clean_ax(ax); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "DIAG_mean_spectra.png"),
            dpi=600, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wavenumbers, band_corr, color="#1f77b4", linewidth=1.0)
ax.axhline( 0.30, color="red",   linestyle="--", linewidth=0.8)
ax.axhline(-0.30, color="red",   linestyle="--", linewidth=0.8)
ax.axhline( 0,    color="black", linewidth=0.5)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Pearson r with target")
ax.set_title("Band-wise Correlation with Target Property")
clean_ax(ax); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "DIAG_band_correlation.png"),
            dpi=600, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(nc_range, rmsecv_list, marker="o", markersize=4,
        color="#1f77b4", linewidth=1.2)
ax.scatter([best_nc], [rmsecv_list[best_pos]], color="red", s=60, zorder=6)
ax.set_xlabel("PLS Components"); ax.set_ylabel("RMSECV")
ax.set_title("PLSR Component Selection (5-Fold CV)")
clean_ax(ax); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "DIAG_PLS_RMSECV_vs_components.png"),
            dpi=600, bbox_inches="tight"); plt.close()

pd.DataFrame({"n_components": nc_range,
              "Train_R2":     train_r2_list,
              "RMSECV":       rmsecv_list}).to_csv(
    os.path.join(CSV_DIR, "DIAG_PLS_RMSECV_vs_components.csv"), index=False)
print("Diagnostic plots saved.\n")

# ==============================================================
# METRICS
# ==============================================================
def calc_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    rpd  = np.std(y_true, ddof=1) / rmse
    rpiq = (np.percentile(y_true, 75) - np.percentile(y_true, 25)) / rmse
    mbe  = np.mean(y_pred - y_true)
    return r2, rmse, mae, rpd, rpiq, mbe

# ==============================================================
# FEATURE SELECTION HELPERS
# ==============================================================
def save_selected(idx, name):
    idx = np.asarray(idx)
    pd.DataFrame({"Index": idx,
                  "Wavenumber": wavenumbers[idx]}).to_csv(
        os.path.join(FEAT_DIR, f"{name}_Selected_Features.csv"), index=False)

def save_selected_vars_plot(idx, tag):
    idx = np.asarray(idx)
    ms  = X_snv.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavenumbers, ms, color="#1f77b4", linewidth=1.0)
    ax.scatter(wavenumbers[idx], ms[idx], c="red", s=10, zorder=5,
               label=f"Selected ({len(idx)})")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity")
    ax.set_title(f"Selected Variables – {tag}")
    ax.legend(fontsize=9); clean_ax(ax); plt.tight_layout()
    fname = f"FS_{tag}_SelectedVariables.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=600, bbox_inches="tight")
    plt.close(); print(f"  Saved: {fname}")

def save_cars_rmsecv_plot(rmsecv_list, best_iter, best_rmse):
    iters = list(range(1, len(rmsecv_list) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, rmsecv_list, marker="o", markersize=2,
            linewidth=0.8, color="#1f77b4")
    ax.scatter([best_iter], [best_rmse], color="red", s=40, zorder=6)
    ax.set_xlabel("Monte Carlo Runs"); ax.set_ylabel("RMSECV")
    ax.set_title(f"Best MC run: {best_iter}  (RMSECV: {best_rmse:.4f})")
    clean_ax(ax); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "FS_CARS_RMSECV.png"),
                dpi=600, bbox_inches="tight"); plt.close()
    print("  Saved: FS_CARS_RMSECV.png")

def save_cars_nvar_rmsecv_plot(nvar_list, rmsecv_list, best_nvar, best_rmse):
    pairs = sorted(zip(nvar_list, rmsecv_list), key=lambda x: x[0])
    xv, yv = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xv, yv, marker="o", markersize=2, linewidth=0.9, color="#1f77b4")
    ax.scatter([best_nvar], [best_rmse], color="red", s=40, zorder=6)
    ax.set_xlabel("Number of Variables"); ax.set_ylabel("RMSECV")
    ax.set_title(f"Selected variables: {best_nvar}  (RMSECV: {best_rmse:.4f})")
    clean_ax(ax); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "FS_CARS_NumVariables_RMSECV.png"),
                dpi=600, bbox_inches="tight"); plt.close()
    print("  Saved: FS_CARS_NumVariables_RMSECV.png")

def save_rmsecv_nvar_plot(nvar_list, rmsecv_list, best_n, best_rmse, tag):
    pairs = sorted(zip(nvar_list, rmsecv_list), key=lambda x: x[0])
    xv, yv = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xv, yv, marker="o", markersize=2, linewidth=0.9, color="#1f77b4")
    ax.scatter([best_n], [best_rmse], color="red", s=40, zorder=6)
    ax.set_xlabel("Number of Variables"); ax.set_ylabel("RMSECV")
    ax.set_title(f"Selected variables: {best_n}  (RMSECV: {best_rmse:.4f})")
    clean_ax(ax); plt.tight_layout()
    fname = f"FS_{tag}_RMSECV.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=600, bbox_inches="tight")
    plt.close(); print(f"  Saved: {fname}")

def save_sa_rmsecv_plot(sa_nvars, sa_best_rmses, best_sa_pos, best_rmse_sa):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sa_nvars, sa_best_rmses, marker="o", markersize=4,
            linewidth=0.9, color="#1f77b4")
    ax.scatter([sa_nvars[best_sa_pos]], [best_rmse_sa],
               color="red", s=40, zorder=6)
    ax.set_xlabel("Number of Variables"); ax.set_ylabel("RMSECV")
    ax.set_title(f"Selected variables: {sa_nvars[best_sa_pos]}"
                 f"  (RMSECV: {best_rmse_sa:.4f})")
    clean_ax(ax); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "FS_SA_RMSECV.png"),
                dpi=600, bbox_inches="tight"); plt.close()
    print("  Saved: FS_SA_RMSECV.png")

# ==============================================================
# MODELS
# ==============================================================
models = {
    "PLSR": lambda p: PLSRegression(
                        n_components=min(best_nc, max(2, p))),
    "SVR":  lambda p: SVR(C=8, gamma="scale", epsilon=0.5),
    "RF":   lambda p: RandomForestRegressor(
                        n_estimators=150, random_state=42,
                        max_features="sqrt", min_samples_leaf=2),
    "XGB":  lambda p: XGBRegressor(
                        n_estimators=150, max_depth=3,
                        learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42, verbosity=0)
}

# ==============================================================
# EVALUATE PIPELINE
# ==============================================================
def evaluate_pipeline(X_tr, X_te, y_tr, y_te,
                      id_te, idx, tag, apply_scaler=False):
    idx = np.asarray(idx)
    tr_rows, te_rows = [], []
    for name, fn in models.items():
        if apply_scaler and name in ["PLSR", "SVR"]:
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X_tr[:, idx])
            Xte = sc.transform(X_te[:, idx])
        else:
            Xtr = X_tr[:, idx]; Xte = X_te[:, idx]
        mdl = fn(len(idx))
        mdl.fit(Xtr, y_tr)
        yp_tr = mdl.predict(Xtr).ravel()
        yp_te = mdl.predict(Xte).ravel()
        pd.DataFrame({"ID": id_te, "Observed": y_te,
                      "Predicted": yp_te}).to_csv(
            os.path.join(CSV_DIR, f"ObsPred_{tag}_{name}.csv"), index=False)
        r2, rmse, mae, rpd, rpiq, mbe = calc_metrics(y_tr, yp_tr)
        tr_rows.append([tag, name, len(idx), r2, rmse, mae, rpd, rpiq, mbe])
        r2, rmse, mae, rpd, rpiq, mbe = calc_metrics(y_te, yp_te)
        te_rows.append([tag, name, len(idx), r2, rmse, mae, rpd, rpiq, mbe])
    return tr_rows, te_rows

# ==============================================================
# SCENARIO 1 — RAW SPECTRA, NO FEATURE SELECTION
# ==============================================================
print("Scenario 1: Raw spectra, No FS...")
tr1, te1 = evaluate_pipeline(
    X_train_raw, X_test_raw, y_train, y_test, ID_test,
    np.arange(X_train_raw.shape[1]), "Raw_NoFS", apply_scaler=True)

# ==============================================================
# SCENARIO 2 — PREPROCESSED, NO FEATURE SELECTION
# ==============================================================
print("Scenario 2: Preprocessed (ALS>UV>SG2>SNV), No FS...")
tr2, te2 = evaluate_pipeline(
    X_train, X_test, y_train, y_test, ID_test,
    np.arange(X_train.shape[1]), "Prep_NoFS", apply_scaler=False)

# ==============================================================
# SCENARIO 3 — PREPROCESSED + FEATURE SELECTION
# ==============================================================
print("\nScenario 3: Feature Selection (CARS | SPA | SA)...")

# ----------------------------------------------------------
# CARS — Competitive Adaptive Reweighted Sampling
# Monte Carlo + Exponential Decay Function + ARS
# ----------------------------------------------------------
print("  Running CARS...")
n_mc       = 50
n_features = X_train.shape[1]
a          = np.power(n_features, 1.0 / (n_mc - 1))
k_cars     = np.log(n_features) / (n_mc - 1)

cars_nvar_list = []
cars_rmse_list = []
cars_subsets   = []

for i in tqdm(range(n_mc), desc="  CARS"):
    mc_idx   = np.random.choice(len(y_train), int(0.8 * len(y_train)),
                                replace=True)
    pls_mc   = PLSRegression(n_components=min(best_nc, X_train.shape[1]))
    pls_mc.fit(X_train[mc_idx], y_train[mc_idx])
    coef_abs = np.abs(pls_mc.coef_.ravel())
    if coef_abs.sum() == 0:
        continue
    weights  = coef_abs / coef_abs.sum()
    keep_n   = max(10, int(np.round(a * np.exp(-k_cars * i) * n_features)))
    valid    = np.where(weights > 0)[0]
    if len(valid) < 10:
        continue
    keep_n   = min(keep_n, len(valid))
    idx_sub  = np.random.choice(valid, size=keep_n, replace=False,
                                p=weights[valid] / weights[valid].sum())
    ycv      = cross_val_predict(
                   PLSRegression(n_components=min(best_nc, keep_n)),
                   X_train[:, idx_sub], y_train, cv=kf)
    rmse     = np.sqrt(mean_squared_error(y_train, ycv))
    cars_nvar_list.append(keep_n)
    cars_rmse_list.append(rmse)
    cars_subsets.append(idx_sub)

best_pos       = int(np.argmin(cars_rmse_list))
cars_idx       = np.array(cars_subsets[best_pos])
best_rmse_cars = cars_rmse_list[best_pos]
print(f"  CARS → {len(cars_idx)} variables  RMSECV: {best_rmse_cars:.4f}")

save_selected(cars_idx, "CARS")
save_selected_vars_plot(cars_idx, "CARS")
save_cars_rmsecv_plot(cars_rmse_list, best_pos + 1, best_rmse_cars)
save_cars_nvar_rmsecv_plot(cars_nvar_list, cars_rmse_list,
                            len(cars_idx), best_rmse_cars)

# ----------------------------------------------------------
# SPA — Successive Projections Algorithm
# ----------------------------------------------------------
print("  Running SPA...")
n_features = X_train.shape[1]
max_vars   = 50

corr      = np.abs([np.corrcoef(X_train[:, i], y_train)[0, 1]
                    for i in range(n_features)])
start_var = int(np.argmax(corr))
spa_chain = [start_var]

for _ in tqdm(range(1, max_vars), desc="  SPA build"):
    remaining = list(set(range(n_features)) - set(spa_chain))
    Xsel      = X_train[:, spa_chain]
    best_var  = None
    best_proj = -np.inf
    for v in remaining:
        x      = X_train[:, v]
        P      = Xsel @ np.linalg.pinv(Xsel)
        proj   = x - P @ x
        proj_n = np.linalg.norm(proj)
        if proj_n > best_proj:
            best_proj = proj_n; best_var = v
    spa_chain.append(best_var)

spa_nvar_list = []
spa_rmse_list = []
spa_subsets   = []

for nv in tqdm(range(2, max_vars + 1), desc="  SPA eval"):
    idx_sub = spa_chain[:nv]
    ycv     = cross_val_predict(
                  PLSRegression(n_components=min(best_nc, nv)),
                  X_train[:, idx_sub], y_train, cv=kf)
    rmse    = np.sqrt(mean_squared_error(y_train, ycv))
    spa_nvar_list.append(nv)
    spa_rmse_list.append(rmse)
    spa_subsets.append(idx_sub)

# 1-SE rule (minimum 10 variables)
min_rmse  = min(spa_rmse_list)
rmse_se   = np.std(spa_rmse_list, ddof=1) / np.sqrt(len(spa_rmse_list))
threshold = min_rmse + rmse_se
valid_pos = [i for i, nv in enumerate(spa_nvar_list)
             if spa_rmse_list[i] <= threshold and nv >= 10]
if len(valid_pos) > 0:
    best_pos = valid_pos[0]
else:
    valid_pos = [i for i, nv in enumerate(spa_nvar_list) if nv >= 10]
    best_pos  = valid_pos[np.argmin([spa_rmse_list[i] for i in valid_pos])]

spa_best      = np.array(spa_subsets[best_pos])
best_rmse_spa = spa_rmse_list[best_pos]
print(f"  SPA → {len(spa_best)} variables  RMSECV: {best_rmse_spa:.4f}")

save_selected(spa_best, "SPA")
save_selected_vars_plot(spa_best, "SPA")
save_rmsecv_nvar_plot(spa_nvar_list, spa_rmse_list,
                      len(spa_best), best_rmse_spa, "SPA")

# ----------------------------------------------------------
# SA — Simulated Annealing (multi-subset-size)
# ----------------------------------------------------------
print("  Running SA...")
sa_nvars      = [10, 15, 20, 25, 30, 35, 40, 45, 50]
sa_best_rmses = []
sa_best_idxs  = []

for nv in sa_nvars:
    curr = np.random.choice(X_train.shape[1], nv, replace=False)
    ycv  = cross_val_predict(
               PLSRegression(n_components=min(best_nc, nv)),
               X_train[:, curr], y_train, cv=kf)
    curr_rmse = np.sqrt(mean_squared_error(y_train, ycv))
    best_r    = curr_rmse
    best_idx  = curr.copy()

    T = 1.0; Tmin = 1e-4; alpha = 0.95
    while T > Tmin:
        trial = curr.copy()
        trial[np.random.randint(nv)] = np.random.randint(X_train.shape[1])
        trial = np.unique(trial)
        if len(trial) < nv:
            missing = np.random.choice(
                list(set(range(X_train.shape[1])) - set(trial)),
                nv - len(trial), replace=False)
            trial = np.concatenate([trial, missing])
        ycv    = cross_val_predict(
                     PLSRegression(n_components=min(best_nc, nv)),
                     X_train[:, trial], y_train, cv=kf)
        t_rmse = np.sqrt(mean_squared_error(y_train, ycv))
        delta  = t_rmse - curr_rmse
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            curr = trial.copy(); curr_rmse = t_rmse
        if curr_rmse < best_r:
            best_r = curr_rmse; best_idx = curr.copy()
        T *= alpha

    sa_best_rmses.append(best_r)
    sa_best_idxs.append(best_idx)
    print(f"    nv={nv:2d}  best RMSECV={best_r:.4f}")

best_sa_pos  = int(np.argmin(sa_best_rmses))
sa_best      = sa_best_idxs[best_sa_pos]
best_rmse_sa = sa_best_rmses[best_sa_pos]
print(f"  SA → {len(sa_best)} variables  RMSECV: {best_rmse_sa:.4f}")

save_selected(sa_best, "SA")
save_selected_vars_plot(sa_best, "SA")
save_sa_rmsecv_plot(sa_nvars, sa_best_rmses, best_sa_pos, best_rmse_sa)

# Evaluate all three FS methods
tr3, te3 = [], []
for idx_fs, tag_fs in [(cars_idx,           "CARS"),
                        (np.array(spa_best), "SPA"),
                        (sa_best,            "SA")]:
    r_tr, r_te = evaluate_pipeline(
        X_train, X_test, y_train, y_test, ID_test,
        idx_fs, tag_fs, apply_scaler=False)
    tr3.extend(r_tr); te3.extend(r_te)

# ==============================================================
# EXPORT ALL RESULTS
# ==============================================================
print("\nExporting results...")
cols = ["Approach", "Model", "Num_Variables",
        "R2", "RMSE", "MAE", "RPD", "RPIQ", "MBE"]

all_tr_dfs = [pd.DataFrame(tr1, columns=cols),
              pd.DataFrame(tr2, columns=cols),
              pd.DataFrame(tr3, columns=cols)]
all_te_dfs = [pd.DataFrame(te1, columns=cols),
              pd.DataFrame(te2, columns=cols),
              pd.DataFrame(te3, columns=cols)]

excel_path = os.path.join(BASE_DIR, "Pipeline_Comparison.xlsx")
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    all_te_dfs[0].to_excel(writer, sheet_name="1_Raw_NoFS",  index=False)
    all_te_dfs[1].to_excel(writer, sheet_name="2_Prep_NoFS", index=False)
    all_te_dfs[2].to_excel(writer, sheet_name="3_Prep_FS",   index=False)

pd.concat(all_tr_dfs, ignore_index=True).to_csv(
    os.path.join(CSV_DIR, "FINAL_Model_Comparison_TRAIN.csv"), index=False)
pd.concat(all_te_dfs, ignore_index=True).to_csv(
    os.path.join(CSV_DIR, "FINAL_Model_Comparison_TEST.csv"), index=False)

pd.DataFrame({
    "Sample_ID": np.concatenate([ID_train, ID_test]),
    "Split":     ["Train"] * len(ID_train) + ["Test"] * len(ID_test),
    "Index":     np.concatenate([train_idx, test_idx])
}).to_csv(os.path.join(CSV_DIR, "Split_Indices.csv"), index=False)

print(f"""
==============================================
PIPELINE COMPLETE
==============================================
PLOTS  → {PLOT_DIR}
  Preprocessing (5) : 01–05_Spectrum_*.png
  Diagnostics   (4) : DIAG_*.png
  Feature Sel.  (7) : FS_CARS/SPA/SA_*.png

CSVs   → {CSV_DIR}
  Preprocessed_Spectra.csv
  Split_Indices.csv
  DIAG_PLS_RMSECV_vs_components.csv
  FINAL_Model_Comparison_TRAIN.csv
  FINAL_Model_Comparison_TEST.csv
  ObsPred_{{tag}}_{{model}}.csv  (12 files)

FEATURES → {FEAT_DIR}
  CARS/SPA/SA _Selected_Features.csv (3 files)

EXCEL  → {excel_path}
==============================================
""")
