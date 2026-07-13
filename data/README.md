# Data Directory

## Real data not included

The actual hyperspectral dataset used in the associated research cannot be shared publicly.

## Running a demo

Use the synthetic data generator to create a test CSV that mirrors the real data structure:

```bash
python data/generate\\\_sample\\\_data.py
```

This creates `data/sample\\\_data.csv` — then update `INPUT\\\_FILE` in `pipeline.py`:

```python
INPUT\\\_FILE = "data/sample\\\_data.csv"
```

## Expected input format

Your CSV must follow this column layout:

|Column position|Content|
|-|-|
|0 (first)|Sample ID (string)|
|1 … N-1|Spectral bands — column headers = wavelength in nm (float)|
|N (last)|Target soil property (float)|

Example (3 bands, 1 target):

```
Sample\\\_ID,400.0,402.5,405.0,Target
S001,0.213,0.198,0.245,3.12
S002,0.189,0.211,0.231,2.87
```

## Notes

* Band columns must be numeric (wavelength values as floats).
* The pipeline auto-detects the number of bands from column count.
* Any continuous soil property can be used as the target (SOC, phosphorus, nitrogen, etc.).

