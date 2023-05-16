import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_leaf_samples=2, max_depth=0):
        self.min_leaf_samples = min_leaf_samples
        self.max_depth = max_depth

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -sum([p * math.log(p) for p in proportions if p > 0])
        return entropy

    def _information_gain(self, y, subset1, subset2):
        n = len(y)
        parent_entropy = self._entropy(y)
        subset1_entropy = self._entropy(subset1)
        subset2_entropy = self._entropy(subset2)

        ig = parent_entropy - \
            (len(subset1) / n * subset1_entropy +
             len(subset2) / n * subset2_entropy)

        return ig

    def _split(self, X, feature, threshold):
        left_idx = np.argwhere(np.array(X)[:, feature] < threshold).flatten()
        right_idx = np.argwhere(np.array(X)[:, feature] >= threshold).flatten()
        return left_idx, right_idx

    def _get_best_split(self, X, y):
        best_split = {"score": -1, "feature": -1, "threshold": -1}

        features = [i for i in range(np.shape(X)[1])]
        for feature in features:
            thresholds = np.unique(np.array(X)[:, feature])

            for threshold in thresholds:
                left_idx, right_idx = self._split(X, feature, threshold)
                subset1 = [y[i] for i in left_idx]
                subset2 = [y[i] for i in right_idx]
                ig = self._information_gain(y, subset1, subset2)
                if (ig > best_split["score"]):
                    best_split["score"] = ig
                    best_split["feature"] = feature
                    best_split["threshold"] = threshold

        return best_split["feature"], best_split["threshold"]

    def _build_tree(self, node, X, y, depth):
        if (depth == 0 or len(X) < self.min_leaf_samples or len(np.unique(y)) <= 1):
            most_common_Label = np.argmax(np.bincount(y))
            node.value = most_common_Label
            return

        feature, threshold = self._get_best_split(X, y)
        node.feature = feature
        node.threshold = threshold
        left_idx, right_idx = self._split(X, feature, threshold)

        node.left = Node()
        X_left = [X[i] for i in left_idx]
        y_left = [y[i] for i in left_idx]
        self._build_tree(node.left, X_left, y_left, depth - 1)

        node.right = Node()
        X_right = [X[i] for i in right_idx]
        y_right = [y[i] for i in right_idx]
        self._build_tree(node.right, X_right, y_right, depth - 1)

    def fit(self, X, y):
        self.root = Node()
        self._build_tree(self.root, X, y, self.max_depth)

    def _traverse(self, x, node):
        if (node.is_leaf()):
            return node.value

        if (x[node.feature] < node.threshold):
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def predict(self, X):
        results = []
        for x in X:
            results.append(self._traverse(x, self.root))
        return results


def main():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.sum(y_test == y_pred) / len(y_test)

    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
