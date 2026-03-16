import matplotlib.pyplot as plt
import numpy as np

x_entry = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_entry = [8, 11, 13, 18, 20, 24, 27, 29, 33, 36]

m, b = np.polyfit(x_entry, y_entry, 1)
plt.scatter(x_entry, y_entry, s=60, alpha=1, edgecolors="k")
plt.plot(x_entry, m * np.array(x_entry) + b)


class SimpleLinearRegression:
    def __init__(self):
        self.x_mean = None  # Average of array X
        self.y_mean = None  # Average of array Y

        self.b0 = None  # Intercept
        self.b1 = None  # Slope

        self.X = None  # Array X
        self.Y = None  # Array Y

        self.error = None  # Possible error

    def fit(self, X: list, Y: list) -> dict:
        self.x_mean = sum(X) / len(X)
        self.y_mean = sum(Y) / len(Y)
        numerator, denominator = 0, 0
        for xi, yi in zip(x_entry, y_entry):
            numerator += (xi - self.x_mean) * (yi - self.y_mean)
            denominator += ((xi - self.x_mean) ** 2)
        self.b1 = numerator / denominator
        self.b0 = self.y_mean - self.b1 * self.x_mean
        self.error = self.mse(X, Y) + self.error_measure(self.x_mean, self.y_mean)
        return {'b0': self.b0, 'b1': self.b1, 'error': self.error}

    def predict(self, x: float) -> float:
        return self.b0 + self.b1 * x

    def error_measure(self, x: float, y_true: float) -> float:
        y_hat = self.predict(x)
        return y_true - y_hat

    def mse(self, X: list = None, Y: list = None) -> float:  # Mean Squared Error
        X = X if X is not None else self.X
        Y = Y if Y is not None else self.Y
        n = len(X)
        MSE = sum([(Y[i] - self.predict(X[i])) ** 2 for i in range(n)])
        return (1 / n) * MSE
