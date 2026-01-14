from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_and_train_model(x_train, y_train, epochs=1, batch_size=1):
    #Build
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    #Compile
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Fit
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model