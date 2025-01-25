import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model():
    # Load preprocessed data
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Save trained model
    joblib.dump(model, 'model.joblib')

if __name__ == "__main__":
    train_model()
