import numpy as np
from sklearn.base import BaseEstimator


class GaussianNaiveBayes(BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mean = np.zeros((self.classes_.size, X.shape[1]))
        self.var = np.zeros((self.classes_.size, X.shape[1]))
        self.priors = np.zeros(self.classes_.size)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

        return self

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes_):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes_[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        probs = [self._predict_proba(x) for x in X]
        return np.array(probs)

    def _predict_proba(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes_):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        max_posterior = max(posteriors)
        posteriors = np.exp(posteriors - max_posterior)
        return posteriors / posteriors.sum()
