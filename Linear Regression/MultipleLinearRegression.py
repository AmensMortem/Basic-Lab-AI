import numpy as np
from sklearn.linear_model import LinearRegression

area = [60, 75, 90, 120, 150, 80, 110, 130, 95, 160]
bedrooms = [2, 3, 3, 4, 4, 2, 3, 4, 3, 5]
age = [20, 15, 10, 8, 5, 25, 12, 7, 18, 3]
price = [150, 200, 250, 320, 400, 180, 290, 350, 240, 450]

X = [  # Each row in X = [Area, Bedrooms, Age]
    [60, 2, 20],
    [75, 3, 15],
    [90, 3, 10],
    [120, 4, 8],
    [150, 4, 5],
    [80, 2, 25],
    [110, 3, 12],
    [130, 4, 7],
    [95, 3, 18],
    [160, 5, 3]
]

y = [150, 200, 250, 320, 400, 180, 290, 350, 240, 450]  # y = Price (in $1000)
x, y = np.array(X), np.array(y)


class SimpleLinearRegression:
    def __init__(self):
        self.x_mean = 0  # Average of array X
        self.y_mean = 0  # Average of array Y

        self.b0 = None  # Intercept
        self.b1 = None  # Slope
        self.sum_b0 = 0

        self.X = None  # Array X
        self.Y = None  # Array Y

        self.price = None
        self.intercept_ = self.b0
        self.coef_ = self.b1
        self.error = None  # Possible error

    def fit(self, X: list, Y: list) -> dict:
        self.y_mean = sum(Y) / len(Y)
        numerator, denominator = 0, 0
        bi = 0
        for x_un in X:
            xi_mean = sum(x_un) / len(x_un)
            self.x_mean += xi_mean
            ni, di = 0, 0
            for xi, yi in zip(x_un, Y):
                ni += (xi - xi_mean) * (yi - self.y_mean)
                di += ((xi - xi_mean) ** 2)
                numerator += (xi - self.x_mean) * (yi - self.y_mean)
                denominator += ((xi - self.x_mean) ** 2)
            bi += ni / di
            self.sum_b0 += bi * xi_mean
        self.b1 = numerator / denominator
        self.b0 = self.y_mean - self.sum_b0
        self.error = self.mse(X, Y) + self.error_measure(self.x_mean, self.y_mean)
        return {'b0': self.b0, 'b1': self.b1, 'error': self.error, 'Y':self.predict(0)}

    def predict(self, x) -> float:
        return self.b0 + self.sum_b0

    def error_measure(self, x: float, y_true: float) -> float:
        y_hat = self.predict(x)
        return y_true - y_hat

    def mse(self, X: list = None, Y: list = None) -> float:  # Mean Squared Error
        X = X if X is not None else self.X
        Y = Y if Y is not None else self.Y
        n = len(X)
        MSE = 0
        for x_u in X:
            MSE += sum([(Y[i] - self.predict(x_u[i])) ** 2 for i in range(n)])
        return (1 / n) * MSE

x_test_1 = [[100, 3, 10]]
model = SimpleLinearRegression()
y_pred = model.fit(x_test_1, [150, 200, 250, 320, 400, 180, 290, 350, 240, 450])
print(y_pred)