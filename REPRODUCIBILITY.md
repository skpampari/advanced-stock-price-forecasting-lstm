# Reproducibility Guide

This repository is structured as a lightweight research artifact for time-series forecasting with reproducible execution and leakage-aware preprocessing.

## Data
Expected file:
- `data/sp500.csv` with columns: `Date`, `Close`

Dataset characteristics from the current run:
- Date range: 1950-01-03 → 2022-09-12
- Total points: 18,292

## Train/Test split
- Train/test split: 80/20 time-ordered (no shuffling)
- Train points: 14,634
- Test points: 3,658
- Sequence length: 60 (60 prior days used to predict the next day)

## Leakage control
- `MinMaxScaler` is fit on training data only (`dataset[:training_data_len]`)
- The fitted scaler is then applied to the full dataset using `scaler.transform(dataset)`

## Random seeds
The run sets fixed seeds for:
- Python `random` (SEED = 42)
- NumPy (SEED = 42)
- TensorFlow (SEED = 42)

Note: numerical results can still vary slightly across different hardware/OS/backends due to non-deterministic low-level kernels. This repo focuses on practical reproducibility: consistent pipeline, methodology, and stable outputs.

## How to run
### Notebook
Open and run:
- `notebooks/exploration.ipynb`

### Scripts (recommended for repeatable runs)
Run:
```bash
python src/main.py
