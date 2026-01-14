# Reproducible LSTM-Based Time Series Forecasting for Financial Index Data

## Overview
This repository provides a reproducible implementation of an LSTM-based time series forecasting pipeline for financial index data (S&P 500). The project is structured as a research-style
artifact, emphasizing proper train/test separation, leakage-free preprocessing, reproducibility, and quantitative evaluation.
This is based on a Master's project in Data Science, focusing on stock price prediction for S&P 500 index data from 1950 to 2022, with extensions for broader financial AI research.

## Why This Matters (PhD/Research Context)

This repository is intended as a research artifact demonstrating:
- clean ML engineering practices for time series,
- correct evaluation methodology (no leakage),
- reproducible experimentation,
- a foundation for future research extensions (baselines, uncertainty, regime shifts).

## Research Scope & Intent
This project intentionally focuses on **univariate time series forecasting** (using closing prices only)
to study temporal dependency modeling with LSTM networks under realistic evaluation constraints.

The goal is not to propose a novel deep learning architecture, but to:
- build a leakage-free and reproducible forecasting pipeline,
- evaluate predictive performance using RMSE,
- establish a baseline for future extensions such as:
  - feature enrichment (volume, technical indicators),
  - baseline model comparisons,
  - uncertainty-aware forecasting,
  - decision-oriented evaluation for financial analytics.


## Methodology
### Data Collection and Preprocessing
- Data Source: CSV file `data/sp500.csv` containing historical S&P 500 data with 'Date' and 'Close' columns.
- Preprocessing Steps:
  - Filter to 'Close' prices.
  - Scale data to [0,1] using MinMaxScaler.
  - Create sequences: 60-day window to predict the next price.
- Challenges Addressed: Data variability (missing values, outliers, fluctuations) via scaling and sequence preparation; Hyperparameter tuning for accuracy; Overfitting/underfitting risks.

### Model Architecture
- LSTM Network: Stacked layers (50 units each), Dense output for regression.
- Optimizer: Adam; Loss: Mean Squared Error (MSE).
- Training: 5 epochs (expandable), batch size 32, on 80% train split.

### Evaluation
- Metrics: Root Mean Squared Error (RMSE), visualization of actual vs. predicted.
- Final Evaluation: Assess accuracy on test set, compare predictions with actual data, identify strengths/weaknesses (e.g., trend capture but volatility issues).

### Key Notes
- Leakage-free scaling (scaler fit on training data only).
- Time-ordered split (no shuffling).
- Fixed seeds for reproducibility.
- Outputs saved to `results/plots/`.
  
## Limitations

- The current implementation uses a univariate input (closing prices only).
- Only simple baseline models (naive forecast) are included; stronger statistical baselines (e.g., moving average, ARIMA) are not yet evaluated..
- Market regime shifts and exogenous variables are not explicitly modeled.
These limitations are intentional and will be addressed in future iterations.
    
## Next Planned Improvements

- Add baseline models (naive forecast, moving average; optional ARIMA)
- Add walk-forward validation (rolling window)
- Add uncertainty estimation (e.g., Monte Carlo dropout)
- Add decision-oriented evaluation (e.g., directional accuracy / cost-aware)

## Installation
1. Clone the repo: `git clone https://github.com/sk/advanced-stock-price-forecasting-lstm.git`
2. Install dependencies: `pip install -r requirements.txt`
   - Contents: numpy, pandas, matplotlib, scikit-learn, tensorflow
3. See `requirements.txt` for pinned dependencies.
     
## Reproducibility Checklist
- Fixed random seeds (Python, NumPy, TensorFlow)
- Train/test split = 80/20 (time-ordered)
- Scaler fit on training portion only (leakage-free)
- Results saved to `results/plots/`

## Usage / Execution Steps
1. **Run Notebook**: Open `notebooks/stock-market-analysis-prediction-using-lstm.ipynb` in Jupyter: `jupyter notebook notebooks/stock-market-analysis-prediction-using-lstm.ipynb`—executes full tutorial (data fetch, EDA, LSTM training, prediction).
2. **Run Scripts**: For modular run:
   - `python src/preprocess.py` (fetches/preprocesses data).
   - `python src/model.py` (builds/trains LSTM).
   - `python src/evaluate.py` (predicts/evaluates/plots).
   - `python src/main.py` (full pipeline).
3. **Outputs**: See `results/` for predictions.csv and plots/actual_vs_predicted.png.

## Results
- Sample RMSE:
| Setting | Value |
|---|---|
| Seed | 42 |
| Split | 80/20 time-ordered |
| Sequence length | 60 |
| Epochs | 5 |
| Batch size | 32 |
| Test RMSE | ~84 |

  Note: RMSE is reported in the original price scale after inverse transformation. Because the S&P 500 index spans a wide numeric range over decades, this metric primarily reflects trend-level
  accuracy; future work will include normalized errors and baseline comparisons for improved interpretability.

- Visualization: Actual vs. predicted curves showing trend alignment.
- Interpretation: RMSE indicates how well the model predicts future S&P 500 prices; lower RMSE means better accuracy.
- This next-step prediction is shown for demonstration and is not a trading strategy.
- Baseline comparison:
  - Naive (t−1) RMSE: reported alongside LSTM for reference.


## References
- Fischer, T., & Krauss, C. (2018). *Deep learning with long short-term memory networks for financial market predictions.* European Journal of Operational Research, 270(2), 654–669.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780.
- Bao, J., Yue, J., & Rao, L. (2017). *A deep learning framework for financial time series using stacked autoencoders and LSTM networks.* PLoS ONE, 12(7), e0180944.

## License
MIT License.

## Contact
Sai Kumar Pampari - skpampari2022@gmail.com - Open to collaborations.



