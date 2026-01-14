from preprocess import load_and_preprocess_data
from model import build_and_train_model
from evaluate import evaluate_and_visualize

if __name__ == "__main__":
    x_train, y_train, scaled_data, training_data_len, scaler, data = load_and_preprocess_data()
    model = build_and_train_model(x_train, y_train)
    evaluate_and_visualize(model, scaled_data, training_data_len, scaler=scaler, data=data)