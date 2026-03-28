from collections import Counter
from sklearn.metrics import euclidean_distances

class KNN:
    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        distance = []
        self.X_train = X
        self.y_train = y
        for num in range(len(X)):
            for check in range(num + 1, len(X)):
                distance.append((self.euclidean_distance(X[num], X[check]))[0][1])

    @staticmethod
    def euclidean_distance(x1, x2):
        return euclidean_distances((x1, x2))

    def predict(self, X_test):
        results = []
        for c in range(len(X_test)):
            distances = []
            for i in range(len(self.X_train)):
                dist = self.euclidean_distance(X_test[c], self.X_train[i])
                distances.append((dist, self.y_train[i]))
            distances.sort(key=lambda x: x[0].sum())
            k_nearest_labels = [label for _, label in distances[:self.k]]
            results.append(Counter(k_nearest_labels).most_common(1)[0][0])
        return results


X_train = [
    [2.1, 1.3, 3.5, 0.5, 1.2],
    [1.8, 1.0, 3.2, 0.4, 1.0],
    [2.5, 1.5, 3.8, 0.6, 1.4],
    [7.2, 6.8, 5.9, 2.1, 3.5],
    [6.9, 6.5, 6.1, 2.3, 3.2],
    [7.5, 7.0, 6.3, 2.0, 3.8],
    [2.0, 1.2, 3.4, 0.5, 1.1],
    [7.1, 6.9, 6.0, 2.2, 3.6]
]

y_train = ["A", "A", "A", "B", "B", "B", "A", "B"]

# -------------------------
# Test Dataset
# -------------------------
X_test = [
    [2.3, 1.4, 3.6, 0.5, 1.3],
    [7.0, 6.7, 6.2, 2.2, 3.4]
]

# -------------------------
# Train and Predict
# -------------------------
knn = KNN(k=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print("Predictions:", predictions)
