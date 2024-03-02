import numpy as np
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
    def __init__(self, min_leaf_samples=2, max_depth=100):
        self.min_leaf_samples = min_leaf_samples
        self.max_depth = max_depth

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        entropy = -np.sum(proportions * np.log(proportions))
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
        left_idx = X[:, feature] < threshold
        right_idx = X[:, feature] >= threshold
        return left_idx, right_idx

    def _get_best_split(self, X, y):
        best_split = {"score": -1, "feature": -1, "threshold": -1}

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_idx, right_idx = self._split(X, feature, threshold)
                subset1 = y[left_idx]
                subset2 = y[right_idx]
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
        X_left = X[left_idx]
        y_left = y[left_idx]
        self._build_tree(node.left, X_left, y_left, depth - 1)

        node.right = Node()
        X_right = X[right_idx]
        y_right = y[right_idx]
        self._build_tree(node.right, X_right, y_right, depth - 1)

    def _traverse(self, x, node):
        node = self.root
        while not node.is_leaf():
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def fit(self, X, y):
        self.root = Node()
        self._build_tree(self.root, X, y, self.max_depth)

    def predict(self, X):
        return [self._traverse(x, self.root) for x in X]


def main():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = np.sum(y_test == y_pred) / len(y_test)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
