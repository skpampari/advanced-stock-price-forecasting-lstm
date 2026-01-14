# Advanced Stock Price Forecasting with LSTM

## Overview
This repository implements a Long Short-Term Memory (LSTM) network for stock market prediction using historical data. It leverages libraries like  Pandas for preprocessing, Keras for model building, and Matplotlib for visualizations. The project explores temporal dependencies in stock data to forecast future prices, including data quality handling, model evaluation with RMSE, and visualizations of actual vs. predicted prices.

This is based on a Master's project in Data Science, focusing on stock price prediction for S&P 500 index data from 1950 to 2022, with extensions for broader financial AI research.

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

### Research Insights
- Achievements:
  - Implementation of LSTM Model: Successful implementation of a machine learning model based on LSTM architecture for time series forecasting.
  - Data Preprocessing: Effective preprocessing of historical stock price data, including feature scaling and sequence preparation for LSTM input.
  - Training and Evaluation: Training the LSTM model on historical data and evaluating its performance using metrics such as Root Mean Squared
    Error (RMSE).
  - Visualization: Visual representation of actual vs predicted stock prices through graphical plots.
- Challenges:
  - Data Quality and Variability: Dealing with the challenges posed by the quality and variability of stock market data, which may include missing
    values, outliers, and sudden market fluctuations.
  - Hyperparameter Tuning: Optimizing hyperparameters of the LSTM model for improved accuracy, considering factors such as the number of LSTM
    units, epochs, and batch size.
  - Overfitting or Underfitting: Addressing the risk of overfitting or underfitting the model to ensure robust performance on both training and
    unseen data.
- Lessons Learned:
  - Complexity of Financial Markets: Understanding the complexity of financial markets and the need for sophisticated models to
    capture their dynamics.
  - Importance of Data Preprocessing: Recognizing the crucial role of data preprocessing in enhancing the model's ability to learn meaningful
    patterns from historical data.
  - Continuous Learning: The stock market is dynamic, and continuous learning and adaptation of models are
    essential to account for changing market conditions.
- Risks:
  - Market Unpredictability: The inherent unpredictability of financial markets, making it challenging to build models that consistently
    outperform the market.
  - Data Limitations: The reliance on historical data and the assumption that past patterns will repeat in the future, which may not always hold
    true in rapidly changing market conditions.
- Benefits:
  - Informed Decision-Making: If successful, the model could provide valuable insights for investors and traders, aiding in more informed decision-
    making.
  - Automation of Analysis: Automation of the stock prediction process, saving time and effort compared to manual analysis.
  - Educational Value: The project serves as an educational tool for understanding machine learning applications in finance.

In conclusion, the Stock Market Prediction using Machine Learning LSTM project aims to harness the power of LSTM networks to enhance stock price forecasting. It involves overcoming challenges related to data quality, model complexity, and market dynamics while deriving valuable insights and lessons from the process. Continuous improvement and adaptation are essential to navigate the complexities.

## Installation
1. Clone the repo: `git clone https://github.com/[yourusername]/advanced-stock-price-forecasting-lstm.git`
2. Install dependencies: `pip install -r requirements.txt`
   - Contents: numpy, pandas, matplotlib, scikit-learn, keras

## Usage / Execution Steps
1. **Run Notebook**: Open `notebooks/stock-market-analysis-prediction-using-lstm.ipynb` in Jupyter: `jupyter notebook notebooks/stock-market-analysis-prediction-using-lstm.ipynb`—executes full tutorial (data fetch, EDA, LSTM training, prediction).
2. **Run Scripts**: For modular run:
   - `python src/preprocess.py` (fetches/preprocesses data).
   - `python src/model.py` (builds/trains LSTM).
   - `python src/evaluate.py` (predicts/evaluates/plots).
   - `python src/main.py` (full pipeline).
3. **Outputs**: See `results/` for predictions.csv and plots/actual_vs_predicted.png.

## Results
- Sample RMSE: ~84 on test set (S&P 500).
- Visualization: Actual vs. predicted curves showing trend alignment.
- Interpretation: RMSE indicates how well the model predicts future S&P 500 prices; lower RMSE means better accuracy.

## References
- Fischer, T. & Krauss, C. (2018). “Deep learning with long short-term memory networks for financial market predictions.” European Journal of
  Operational Research, 270(2), 654-669
- Hochreiter & Schmidhuber, 1997. “Long Short-Term Memory.”
- Zhang et al., 2017. “Stock Price Prediction Using LSTM.”

## License
MIT License.

## Contact
Sai Kumar Pampari - skpampari2022@gmail.com - Open to collaborations.


