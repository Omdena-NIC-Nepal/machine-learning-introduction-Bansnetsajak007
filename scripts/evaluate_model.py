import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model():
    # Loadi preprocessed data and predictions
    X_test_scaled = np.load('../data/X_test_scaled.npy')
    y_test = np.load('../data/y_test.npy')
    y_pred = np.load('../data/y_pred.npy')

    # Residual plot visualization
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

if __name__ == "__main__":
    evaluate_model()