from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score

from gaussian_naive_bayes import GaussianNaiveBayes

import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target

gnb = GaussianNaiveBayes()

skf = StratifiedKFold(n_splits=5, shuffle=True)
accuracies = cross_val_score(gnb, X, y, cv=skf, scoring="accuracy")
precisions = cross_val_score(gnb, X, y, cv=skf, scoring="precision_macro")
recalls = cross_val_score(gnb, X, y, cv=skf, scoring="recall_macro")
f1s = cross_val_score(gnb, X, y, cv=skf, scoring="f1_macro")


df = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1 Score"])

df.loc[0] = [
    "Gaussian Naive Bayes",
    f"{accuracies.mean():.3f} ± {accuracies.std():.3f}",
    f"{precisions.mean():.3f} ± {precisions.std():.3f}",
    f"{recalls.mean():.3f} ± {recalls.std():.3f}",
    f"{f1s.mean():.3f} ± {f1s.std():.3f}",
]

df.loc[1] = [
    "Decission Tree",
    "0.958 +- 0.035",
    "0.961 +- 0.034",
    "0.958 +- 0.035",
    "0.958 +- 0.036",
]

df.loc[2] = [
    "SVM",
    "0.984 +- 0.029",
    "0.986 +- 0.028",
    "0.984 +- 0.029",
    "0.984 +- 0.030",
]

print(df)
df.to_excel("results.xlsx", index=False)
