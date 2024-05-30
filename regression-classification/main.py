from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from typing import Literal, Optional


iris = load_iris()
X = iris.data
y = iris.target

tree_scores = np.empty((0, 4))


def do_measure_tree(
    criterion: Literal["gini", "entropy", "log_loss"],
    splitter: Literal["best", "random"],
    max_depth: Optional[int] = None,
) -> np.array:

    # Kryteria oceny jako średnia z 3*5 wyników +- stdev
    # Tabela kryterium oceny, technika podziału wezla, max glebokosc, acc, prec, recall, F1
    result = np.empty((0, 7))

    for random_state in [42, 63, 71]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Wykres F1(max_depth)
            tree = DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                random_state=random_state,
            )
            tree.fit(X_train, y_train)

            y_pred = tree.predict(X_test)
            result = np.vstack(
                [
                    result,
                    np.array(
                        [
                            criterion,
                            splitter,
                            max_depth,
                            accuracy_score(y_test, y_pred),
                            precision_score(y_test, y_pred, average="weighted"),
                            recall_score(y_test, y_pred, average="weighted"),
                            f1_score(y_test, y_pred, average="weighted"),
                        ]
                    ),
                ]
            )

    numerical_columns = result[:, 3:].astype(float)
    means = np.mean(numerical_columns, axis=0)
    stds = np.std(numerical_columns, axis=0)
    scores = np.array([f"{means[i]:.3f} +- {stds[i]:.3f}" for i in range(len(means))])

    return np.hstack(
        [
            np.array([criterion, splitter, max_depth if max_depth != None else "auto"]),
            scores,
        ]
    )


def do_measure_svm(
    C: float = 1,
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
    max_iter: int = ...,
) -> np.array:

    # Tabela siła regularyzacji C, funkcja jądra kernel, l. iteracji max_iter (dla badania kernela i C moze byc auto), acc, prec, recall, F1
    # Wykresy for each krenel F1(C)
    result = np.empty((0, 7))

    for random_state in [42, 63, 71]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            svm = (
                SVC(
                    C=C,
                    kernel=kernel,
                    max_iter=max_iter,
                    random_state=random_state,
                    tol=10e-9,
                )
                if max_iter != None
                else SVC(C=C, kernel=kernel, random_state=random_state, tol=10e-9)
            )
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)

            result = np.vstack(
                [
                    result,
                    np.array(
                        [
                            kernel,
                            C,
                            max_iter,
                            accuracy_score(y_test, y_pred),
                            precision_score(y_test, y_pred, average="weighted"),
                            recall_score(y_test, y_pred, average="weighted"),
                            f1_score(y_test, y_pred, average="weighted"),
                        ]
                    ),
                ]
            )

    numerical_columns = result[:, 3:].astype(float)
    means = np.mean(numerical_columns, axis=0)
    stds = np.std(numerical_columns, axis=0)
    scores = np.array([f"{means[i]:.3f} +- {stds[i]:.3f}" for i in range(len(means))])

    return np.hstack(
        [
            np.array([kernel, C, max_iter if max_iter != None else "auto"]),
            scores,
        ]
    )


results = np.empty((0, 7))
for criterion in ["gini", "entropy", "log_loss"]:
    for splitter in ["best", "random"]:
        for max_depth in [5, 10, None]:
            results = np.vstack(
                [
                    results,
                    do_measure_tree(
                        criterion=criterion, splitter=splitter, max_depth=max_depth
                    ),
                ]
            )

df = pd.DataFrame(
    results,
    columns=[
        "criterion",
        "splitter",
        "max_depth",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)

print(df)
# df.to_excel("DecisionTree.xlsx", index=False)


results = np.empty((0, 7))
for kernel in ["linear", "poly", "rbf"]:
    for C in [0.01, 0.1, 1]:
        for max_iter in [1, 10, 100, None]:
            row = do_measure_svm(C=C, kernel=kernel, max_iter=max_iter)
            print(row)
            results = np.vstack(
                [
                    results,
                    row,
                ]
            )

df = pd.DataFrame(
    results,
    columns=[
        "kernel",
        "C",
        "max_iter",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)

print(df)
# df.to_excel("SVM.xlsx", index=False)


def plot_relations(kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]):

    # Tabela siła regularyzacji C, funkcja jądra kernel, l. iteracji max_iter (dla badania kernela i C moze byc auto), acc, prec, recall, F1
    # Wykresy for each krenel F1(C)
    C = np.logspace(-3, 3, 7)  # Use logarithmic scale for C values
    f1 = []

    for c in C:

        all_f1 = []
        for random_state in [42, 63, 71]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                svm = SVC(
                    C=c,
                    kernel=kernel,
                )

                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                all_f1.append(
                    f1_score(y_test, y_pred, average="weighted"),
                )

        f1.append(np.mean(all_f1))
        all_f1 = []

    # Plot F1 scores against C values
    plt.figure()
    plt.plot(C, f1, marker="o")
    plt.xscale("log")  # Set logarithmic scale for x-axis
    plt.xlabel("Regularization parameter C")
    plt.ylabel("F1 Score")
    plt.title("F1(C) - " + str(kernel))
    plt.grid(True)
    plt.savefig(f"{kernel}.png")
    plt.show()


for kernel in ["linear", "poly", "rbf"]:
    plot_relations(kernel)
