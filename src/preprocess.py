import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import os
import pandas as pd
import matplotlib.pyplot as plt



def load_and_preprocess_data(
    csv_path="data/sp500.csv",
    seq_length=60):   
    # Load CSV
    df = pd.read_csv(csv_path)
    # Parse and sort dates
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    #Shape (for logging)
    print(df.shape)
    #Visualize (optional, comment out for script)
    plt.figure(figsize=(16,8)); plt.title('Close Price History'); plt.plot(df['Close']); plt.xlabel('Date', fontsize=18); plt.ylabel('Close Price USD ($)', fontsize=18); plt.show()
    # Filter Close
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)
    #Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    #Train data/sequences
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(seq_length, len(train_data)):
        x_train.append(train_data[i-seq_length:i, 0])
        y_train.append(train_data[i, 0])
    #To arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaled_data, training_data_len, scaler, data