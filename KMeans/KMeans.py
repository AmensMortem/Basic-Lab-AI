import numpy as np


class KMeans:
    """
    K-Means clustering with K-Means++ initialization.
    Parameters
    ----------
    k : int
        Number of clusters.
    max_iters : int
        Maximum number of iterations (default: 300).
    tol : float
        Convergence tolerance — stops when centroid shift is below this value (default: 1e-4).
    random_state : int or None
        Seed for reproducibility (default: None).
    """

    def __init__(self, k: int = 3, max_iters: int = 300, tol: float = 1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iters_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Compute K-Means clustering on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self._validate(X)
        if X.shape[0] < self.k:
            raise ValueError
        self.centroids_ = self._init_centroids(X, np.random.default_rng(self.random_state))
        for i in range(self.max_iters):
            self.n_iters_ = i + 1

            labels = self._assign(X)
            new_centroids = self._update(X, labels)
            self.centroids_ = new_centroids

            shift = np.linalg.norm(new_centroids - self.centroids_)
            if shift < self.tol:  # Error tolerance test
                break
        self.labels_ = self._assign(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample in X to the nearest centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self._check_fitted()
        X = self._validate(X)
        return self._assign(X)


    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels for X."""
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """K-Means++ initialization for smarter starting centroids."""
        n_samples, n_features = X.shape
        centroids = np.empty((self.k, n_features))

        initial_idx = rng.integers(n_samples)
        centroids[0] = X[initial_idx]

        for i in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]]) for x in X])
            prob = distances / distances.sum()

            next_idx = rng.choice(n_samples, p=prob)
            centroids[i] = X[next_idx]

        return centroids

    def _assign(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to its nearest centroid (E-step)."""
        # Euclidean distances: (n_samples, k)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recompute centroids as the mean of assigned points (M-step)."""
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = self.centroids_[i]
        return new_centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Sum of squared distances from each point to its assigned centroid."""
        distances = X - self.centroids_[labels]
        return float((distances ** 2).sum())

    @staticmethod
    def _validate(X) -> np.ndarray:
        X = np.array(X, dtype=float)
        if X.ndim != 2:
            raise ValueError('2 dimensions to cluster')

        if X.shape[0] == 0:
            raise ValueError("Empty")
        return X

    def _check_fitted(self):
        if self.centroids_ is None:
            raise RuntimeError

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"KMeans(k={self.k}, max_iters={self.max_iters}, "
            f"tol={self.tol}, random_state={self.random_state})"
        )


rng = np.random.default_rng(
    42)  # numpy.random.default_rng is a function in NumPy that creates a new random number generator (RNG) using the Generator class
cluster_1 = rng.normal([0, 0], 0.5, (50, 2))
cluster_2 = rng.normal([4, 0], 0.5, (50, 2))
cluster_3 = rng.normal([2, 4], 0.5, (50, 2))
cluster_4 = rng.normal([2, 1.5], 0.3, (50, 2))

# Three well-separated blobs
X = np.vstack([
    cluster_1,
    cluster_2,
    cluster_3,
    cluster_4
])
y = [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50
model = KMeans(k=4, random_state=42)
model.fit(X)  # KMeans(k=4, max_iters=300, tol=0.0001, random_state=42)
new_points = np.array([[0.1, 0.1], [4.0, 0.2], [2.1, 3.9]])
print(f"\nPredictions for {new_points.tolist()}:")
print(model.predict(new_points))

import matplotlib.pyplot as plt


training_labels = model.labels_# Predictions for the training data and the new points
new_point_predictions = model.predict(new_points)
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=training_labels, cmap='viridis', s=30, alpha=0.5, label='Training Data')
plt.scatter(model.centroids_[:, 0], model.centroids_[:, 1],# final centroids
            c='red', marker='X', s=200, edgecolors='black', label='Centroids')
plt.scatter(new_points[:, 0], new_points[:, 1],# Plot the "new_points" (highlighted as stars)
            c=new_point_predictions, cmap='viridis', marker='*', s=300,
            edgecolors='black', linewidths=2, label='Predicted New Points')
plt.title(f"K-Means Clustering with {model.k} Clusters", fontsize=15)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()