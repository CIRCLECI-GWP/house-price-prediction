import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def evaluate_model():
    # Load preprocessed data and trained model
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    model = joblib.load('model.joblib')
    # Evaluate model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    evaluate_model()
