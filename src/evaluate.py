import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd

def evaluate_and_visualize(model, scaled_data, training_data_len, seq_length=60, scaler=None, data=None):
    #Test data
    test_data = scaled_data[training_data_len - seq_length: , :]
    x_test = []
    y_test = data.values[training_data_len : , :]
    for i in range(seq_length, len(test_data)):
        x_test.append(test_data[i-seq_length:i, 0])
    #To array
    x_test = np.array(x_test)
    #Reshape
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Predict
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    #RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
    print(rmse)
    #Plot
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig('results/plots/actual_vs_predicted.png')
    plt.close()
    #Show valid
    print(valid)
    #Future prediction
    new_df = data.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)
  