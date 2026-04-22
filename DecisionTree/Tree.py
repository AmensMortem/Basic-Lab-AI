import math
import pandas as pd
import numpy as np


class Node:
    def __init__(self, feature=None, children=None, prediction=None, depth=0):
        self.feature = feature
        self.children = children if children is not None else {}
        self.prediction = prediction
        self.depth = depth


class DecisionTreeClassifierScratch:
    def __init__(
            self,
            criterion="entropy",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=None, ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.tree_ = None
        self.classes_ = None
        self.feature_names_in_ = None
        self.target_name_ = None
        self._global_majority_ = None

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def get_params(self, deep=True):
        return {
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            match key:
                case "criterion":
                    self.criterion = value
                case "max_depth":
                    self.max_depth = value
                case "min_samples_split":
                    self.min_samples_split = value
                case "min_samples_leaf":
                    self.min_samples_leaf = value
                case "random_state":
                    self.random_state = value

    # ------------------------------------------------------------------
    # Impurity measures
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(y):
        ids, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()  # probability of the class
        return -np.sum(p * np.log2(p + 1e-12))

    @staticmethod
    def _gini(y):
        ids, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()  # probability of the class
        return 1 - np.sum(p ** 2)

    def _impurity(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        # elif self.criterion == 'entropy':
        return self._entropy(y)
        # else:
        #     raise ValueError('Criterion error')

    # ------------------------------------------------------------------
    # Splitting logic
    # ------------------------------------------------------------------

    def _information_gain(self, data, feature, target):
        parent_impurity = self._impurity(data[target])
        total = len(data)
        weighted_child_impurity = 0.0
        for value, subset in data.groupby(feature, sort=False):
            weight = len(subset) / total
            weighted_child_impurity += weight * self._impurity(subset[target])

        return parent_impurity - weighted_child_impurity

    def _best_split(self, data, features, target):
        best_feature = None
        best_gain = -np.inf
        for feature in features:
            gain = self._information_gain(data, feature, target)
            if best_gain < gain:
                best_gain = gain
                best_feature = feature
        return best_feature

    # ------------------------------------------------------------------
    # Majority class — uses np.unique for robustness across array types
    # ------------------------------------------------------------------

    def _majority_class(self, y):
        values, counts = np.unique(np.asarray(y), return_counts=True)
        max_count = counts.max()
        candidates = values[counts == max_count]
        if len(candidates) == 1:
            return candidates[0]
        rng = np.random.default_rng(self.random_state)
        return candidates[rng.integers(len(candidates))]
        """
        Returns the most frequent class in y.
        Accepts any array-like (list, np.ndarray, pd.Series).
        Uses random_state for reproducible tie-breaking.
        """

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, data, features, target, depth=0):
        y = data[target]

        if y.nunique() == 1:
            return Node(prediction=y.iloc[0], depth=depth)

        if not features:
            return Node(prediction=self._majority_class(y), depth=depth)

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=self._majority_class(y), depth=depth)

        if len(data) < self.min_samples_split:
            return Node(prediction=self._majority_class(y), depth=depth)

        best_feature = self._best_split(data, features, target)
        if not features:
            return Node(prediction=self._majority_class(data[target]), depth=depth)

        children = {}
        remaining_features = [f for f in features if f != best_feature]
        for value, subset in data.groupby(best_feature, sort=False):
            if len(subset) < self.min_samples_leaf:
                children[value] = Node(prediction=self._majority_class(subset[target]), depth=depth + 1)
            else:
                children[value] = self._build_tree(subset, remaining_features, target, depth + 1)
        return Node(feature=best_feature, children=children, depth=depth)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.target_name_ = y.name
        y = pd.Series(y).array
        x = X.reset_index(drop=True)
        self.feature_names_in_ = (list(x.columns))
        self.classes_ = np.unique(y)
        data = X.copy()
        data[self.target_name_] = y
        self._global_majority_ = self._majority_class(y)
        self.tree_ = self._build_tree(data, self.feature_names_in_, self.target_name_)
        return self
        # ROOT FIX: reset both X and y to a clean 0-based index before
        # combining into a single DataFrame, so pandas does not
        # produce NaN rows due to index misalignment (e.g. after
        # train_test_split which preserves the original shuffled index)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _collect_all_leaf_labels(self, node):
        if not node.children:  # leaf
            return [node.prediction]
        labels = []
        for child in node.children.values():
            labels.extend(self._collect_all_leaf_labels(child))
        return labels
        """Recursively collects ALL leaf labels — ensures unbiased majority vote."""


    def _predict_one(self, row, node):
        if node.prediction is not None:
            return node.prediction
        feature_value = row.get(node.feature)
        if feature_value in node.children:
            return self._predict_one(row, node.children[feature_value])
        all_labels = self._collect_all_leaf_labels(node)
        if all_labels:
            return self._majority_class(all_labels)
        return self._global_majority_

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X):
        if self.tree_ is None:
            raise RuntimeError("Call fit() before predict().")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = X.reset_index(drop=True)
        return np.array([self._predict_one(row, self.tree_) for _, row in X.iterrows()])

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self, X, y):
        y = np.asarray(y)
        predictions = self.predict(X)
        return np.mean(predictions == y)

    # ------------------------------------------------------------------
    # Utility: pretty-print tree structure
    # ------------------------------------------------------------------

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree_

        if node.prediction is not None:
            print(f"{indent}→ Predict: {node.prediction}")
        else:
            print(f"{indent}[{node.feature}]")
            children = list(node.children.items())
            for i, (value, child) in enumerate(children):
                print(f"{indent}  == {value}:")
                child_indent = indent + ("    " if i < len(children) - 1 else "    ")
                self.print_tree(child, child_indent)

