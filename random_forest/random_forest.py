import numpy as np
import sys
from pathlib import Path
from scipy import stats
import math
import time

# to add the parent directory to the system path to allow imports
file = Path.cwd()
package_root_directory = file.parents[0]
sys.path.append(str(package_root_directory))

from decision_tree.decision_tree_classifier import MyDecisionTreeClassifier


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


class MyRandomForestClassifier:
    def __init__(
        self, n_estimators=100, min_samples_split=2, max_depth=100, max_features="auto"
    ):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.estimators_ = []

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        n_samples, n_features = X.shape

        if not self.max_features:
            self.max_features = n_features
        elif self.max_features == "auto":
            self.max_features = math.ceil(np.sqrt(n_features))

        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = MyDecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                max_features=self.max_features,
            )
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.estimators_])
        y_pred_majority_votes, n_votes = stats.mode(tree_preds, axis=0)
        return y_pred_majority_votes.flatten()


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    # option 0: hearts dataset; option 1: breast cancer dataset
    DATASET_OPTION = 0
    # option 0: use only seeds 0 and 1; else use seeds from 0 to 40
    USE_ALL_SEEDS = 1

    if DATASET_OPTION == 0:
        print("[INFO] Using UCI hearts dataset")
        df = pd.read_csv("../decision_tree/heart.csv")
        X = df.drop(columns="target")
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        print("[INFO] Using breast cancer dataset")
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    n_trees = 3
    max_depth = 7
    seeds = np.arange(0, 41) if USE_ALL_SEEDS else np.arange(0, 2)
    print(f"[INFO] Using {n_trees} Decision Trees in the forest")

    accuracy_list = []
    time_list = []
    accuracy_list_sk = []
    time_list_sk = []

    for seed in seeds:
        print(f"[INFO] Using random seed {seed}")
        np.random.seed(seed)
        clf = MyRandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
        start_time = time.perf_counter()
        clf.fit(X_train, y_train)
        total_time = time.perf_counter() - start_time

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy_list.append(acc)
        time_list.append(total_time)
        # Much better than the accuracy of a single DecisionTree of about 0.82
        print("My implementation: Accuracy =", acc)
        print(f"Total training time = {total_time:.4f} secs\n")

        sk_clf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=max_depth, random_state=seed
        )
        start_time = time.perf_counter()
        sk_clf.fit(X_train, y_train)
        total_time = time.perf_counter() - start_time

        sk_y_pred = sk_clf.predict(X_test)
        acc = accuracy_score(y_test, sk_y_pred)
        accuracy_list_sk.append(acc)
        time_list_sk.append(total_time)
        print("Sklearn implementation: Accuracy =", acc)
        print(f"Total training time = {total_time:.4f} secs\n")

    print("\nMy implementation")
    print("---" * 8)
    # 0.8101
    print(f"Avg accuracy: {np.mean(accuracy_list):.4f}")
    print(f"Avg training time: {np.mean(time_list):.4f} secs")

    print("\nSklearn implementation")
    print("---" * 8)
    # 0.7981
    print(f"Avg accuracy: {np.mean(accuracy_list_sk):.4f}")
    print(f"Avg training time: {np.mean(time_list_sk):.4f} secs")
