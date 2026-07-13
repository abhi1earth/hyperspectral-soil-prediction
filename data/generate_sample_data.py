# ==============================================================
# Synthetic Sample Data Generator
# ==============================================================
# Generates a small synthetic dataset that mirrors the structure
# of real hyperspectral data so you can test the pipeline
# without any real measurements.
#
# Output : data/sample_data.csv
# Columns: Sample_ID | wavelength_1 ... wavelength_N | Target
# ==============================================================

import numpy as np
import pandas as pd

np.random.seed(42)

# ---- Configuration ----
N_SAMPLES    = 80      # number of spectra
N_BANDS      = 100     # number of spectral bands
WL_START     = 400.0   # starting wavelength (nm)
WL_STEP      = 2.5     # wavelength step (nm)
TARGET_MEAN  = 3.5     # synthetic target mean
TARGET_STD   = 1.2     # synthetic target std

# ---- Wavelength axis ----
wavelengths = [f"{WL_START + i * WL_STEP:.1f}" for i in range(N_BANDS)]

# ---- Simulate spectra with smooth structure ----
# Use a sum of Gaussians to create realistic-looking spectra
x = np.linspace(0, 1, N_BANDS)
base_spec = (0.6 * np.exp(-((x - 0.3) ** 2) / 0.02) +
             0.4 * np.exp(-((x - 0.7) ** 2) / 0.03) +
             0.2 * np.exp(-((x - 0.5) ** 2) / 0.05))

# Add per-sample variation + noise
spectra = np.array([
    base_spec * np.random.uniform(0.7, 1.3) +
    np.random.normal(0, 0.02, N_BANDS)
    for _ in range(N_SAMPLES)
])
spectra = np.clip(spectra, 0, None)    # no negative reflectance

# ---- Generate synthetic target (correlated with a few bands) ----
key_bands = [20, 45, 70]              # bands that "drive" the target
target    = (
    TARGET_MEAN
    + 0.8 * spectra[:, key_bands[0]]
    - 0.5 * spectra[:, key_bands[1]]
    + 0.3 * spectra[:, key_bands[2]]
    + np.random.normal(0, 0.3, N_SAMPLES)
)
# Rescale to realistic range
target = (target - target.mean()) / target.std() * TARGET_STD + TARGET_MEAN
target = np.clip(target, 0.5, 8.0)

# ---- Assemble DataFrame ----
df = pd.DataFrame(spectra, columns=wavelengths)
df.insert(0, "Sample_ID", [f"S{i+1:03d}" for i in range(N_SAMPLES)])
df["Target"] = target.round(4)

# ---- Save ----
out_path = "data/sample_data.csv"
df.to_csv(out_path, index=False)

print(f"Synthetic dataset saved to: {out_path}")
print(f"  Samples : {N_SAMPLES}")
print(f"  Bands   : {N_BANDS}  ({WL_START} – {WL_START + (N_BANDS-1)*WL_STEP:.1f} nm)")
print(f"  Target  : mean={target.mean():.3f}  std={target.std():.3f}  "
      f"min={target.min():.3f}  max={target.max():.3f}")
print("\nUpdate INPUT_FILE in pipeline.py to 'data/sample_data.csv' to run the demo.")
