import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate():
    # Load preprocessed data
    X_train_scaled = np.load('../data/X_train_scaled.npy')
    X_test_scaled = np.load('../data/X_test_scaled.npy')
    y_train = np.load('../data/y_train.npy')
    y_test = np.load('../data/y_test.npy')

    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Save predictions
    np.save('../data/y_pred.npy', y_pred)

if __name__ == "__main__":
    train_and_evaluate()