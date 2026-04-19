# IMU-Terrain-Classification-using-LightGBM-with-Data-Augmentation
Machine learning project for terrain classification using IMU sensor data (gyroscope + accelerometer). Includes custom dataset augmentation with scaling and noise injection, feature engineering, and LightGBM model training for robust terrain recognition across multiple surface conditions.

# IMU Terrain Classification

Machine learning pipeline for classifying terrain type using IMU (Inertial Measurement Unit) sensor data. Two models are included: LightGBM (`light_gbm.ipynb`) and XGBoost (`xg_boost.py`). Both were trained and evaluated on the same dataset with identical hyperparameters for a fair comparison.

---

## Repository Structure

```
IMU DATASET/               - Raw CSV files organized by terrain class
augment_code.py            - Data augmentation script (scaling + noise injection)
augmented_dataset.csv      - Augmented training data
imu_dataset_original.csv   - Original unmodified dataset
light_gbm.ipynb            - LightGBM classifier notebook
xg_boost.py                - XGBoost classifier script
README.md                  - This file
```

---

## Dataset

The IMU dataset contains sensor readings across five terrain classes:

| Label | Class      |
|-------|------------|
| 0     | ASPHALT    |
| 1     | CONCRETE   |
| 2     | DIRT_ROAD  |
| 3     | PLOUGHED   |
| 4     | UNPLOUGHED |

Each raw file contains six sensor channels per time step: `wx`, `wy`, `wz` (gyroscope angular velocities) and `ax`, `ay`, `az` (accelerometer linear accelerations).

The dataset is imbalanced. ASPHALT has significantly more samples than DIRT_ROAD and CONCRETE, which have the fewest windows.

---

## Feature Extraction

A sliding window approach is used to segment the time-series data. Each window produces 43 features:

- 7 statistical features per axis (mean, std, RMS, min, max, skewness, kurtosis) across all 6 axes = 42 features
- 1 physics-inspired feature: mean Z-axis acceleration energy

Window configuration:

```
WINDOW_SIZE = 400   (samples per window)
STRIDE      = 200   (step between windows)
```

---

## Data Augmentation

`augment_code.py` generates synthetic samples by applying two transformations to the original data: amplitude scaling and Gaussian noise injection. The augmented output is saved to `augmented_dataset.csv`. This was used to address class imbalance before model training.

---

## Models

Both models use identical hyperparameters to enable direct comparison:

| Hyperparameter               | Value |
|------------------------------|-------|
| n_estimators                 | 300   |
| max_depth                    | 7     |
| learning_rate                | 0.05  |
| subsample                    | 0.7   |
| colsample / feature_fraction | 0.7   |
| L1 regularization            | 0.1   |
| L2 regularization            | 1.0   |

Train/test split: 80/20 stratified. Feature scaling: StandardScaler (zero mean, unit variance) fit on the training set only.

Cross-validation: 5-fold stratified.

---

## Requirements

```
Python 3.7+
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
```

Install all dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn xgboost lightgbm
```

---

## Usage

### XGBoost

```bash
python xg_boost.py
```

The script expects the dataset folder at `./IMU_DATASET` with subfolders named after each terrain class. Each subfolder should contain CSV files prefixed with `imu_` and columns `wx`, `wy`, `wz`, `ax`, `ay`, `az`.

Outputs generated:

- `confusion_matrix_xgboost.png`
- `feature_importance_top15_xgboost.png`
- `feature_importance_all_xgboost.png`

### LightGBM

Open `light_gbm.ipynb` in Jupyter and run all cells. Same dataset path and output format apply.

---

## Known Issues Fixed

The original XGBoost notebook had two bugs that were corrected in `xg_boost.py`:

1. **Missing closing quote** in `DATASET_PATH = "./IMU_DATASET"` — this caused a JSON parse error that rendered the entire notebook invalid on GitHub.
2. **Global variable leak inside `load_imu_data()`** — the function accepted `window_size` and `stride` as parameters but internally referenced the globals `WINDOW_SIZE` and `STRIDE`, meaning any custom values passed to the function would be silently ignored.

---

## Notes

- DIRT_ROAD and CONCRETE have the fewest training samples due to limited recordings. Collecting more data for these classes would likely improve their classification performance.
- Any CSV file in the dataset with fewer rows than the window size (400) is automatically skipped during loading.
- Cross-validation is performed on the training split only. The test set is held out completely until final evaluation.
