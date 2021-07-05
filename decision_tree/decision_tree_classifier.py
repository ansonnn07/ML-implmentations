import numpy as np
import pandas as pd
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # the indices of the feature at this node
        self.feature = feature
        # the threshold used for splitting at this node
        self.threshold = threshold
        # the left child node
        self.left = left
        # the right child node
        self.right = right
        # the output label at this node based on the largest vote
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class MyDecisionTreeClassifier:
    def __init__(self, max_depth=100, min_samples_split=2, max_features=None):
        # The maximum depth of the tree
        self.max_depth = max_depth
        # The minimum number of samples required to split an internal node
        self.min_samples_split = min_samples_split
        # The number of features to consider when looking for the best split
        self.max_features = max_features
        # the tree to be built, starting from the root node
        self.tree = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        n_samples, n_features = X.shape
        if self.max_features:
            assert self.max_features <= n_features
        else:
            self.max_features = n_features

        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """A recursive function used to build the tree starting from the root node."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y.flatten()))

        # if hit any stopping criteria, return a leaf node
        if (depth >= self.max_depth
                    or n_classes == 1
                    or n_samples < self.min_samples_split
                ):
            # return the output label as the value for the leaf node
            leaf_value = self._predict_class(y)
            return Node(value=leaf_value)

        # randomly choose max_features of features
        feature_idxs = np.random.choice(
            n_features, self.max_features, replace=False)

        # greedy search the best feature and splitting threshold
        best_feat_idxs, best_thresh = self._best_criteria(X, y, feature_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat_idxs], best_thresh)
        left_branch = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_branch = self._grow_tree(
            X[right_idxs, :], y[right_idxs], depth + 1)
        # if not leaf node yet, return a node with no leaf `value` given
        return Node(best_feat_idxs, best_thresh, left_branch, right_branch)

    def _predict_class(self, y):
        """Compute the highest voted class"""
        counts = np.bincount(y)
        return np.argmax(counts)

    def _get_votes(self, y, thresh_idxs):
        y_thresh = y[thresh_idxs]
        votes = np.bincount(y_thresh)
        return votes

    def _get_gini(self, votes):
        total_votes = np.sum(votes)
        gini = 1 - np.sum(np.square(votes / total_votes))
        return gini

    def _split(self, feature_col, threshold):
        left_idxs = np.where(feature_col <= threshold)[0]
        right_idxs = np.where(feature_col > threshold)[0]
        return left_idxs, right_idxs

    def _criterion_value(self, y, feature_col, threshold):
        """Compute the total gini_impurity as the criterion"""
        left_idxs, right_idxs = self._split(feature_col, threshold)

        votes_left, votes_right = self._get_votes(
            y, left_idxs), self._get_votes(y, right_idxs)

        gini_left, gini_right = self._get_gini(
            votes_left), self._get_gini(votes_right)

        total_votes_left = np.sum(votes_left)
        total_votes_right = np.sum(votes_right)
        total_votes_all = total_votes_left + total_votes_right
        total_gini = (total_votes_left / total_votes_all) * gini_left + \
            (total_votes_right / total_votes_all) * gini_right
        return total_gini

    def _best_criteria(self, X, y, feature_idxs):
        best_gini = np.inf
        for feature_idx in feature_idxs:
            feature_col = X[:, feature_idx]
            unique_sorted = np.unique(feature_col)
            # take the midpoints of adjacent sorted unique values as thresholds
            thresholds = (unique_sorted[:-1] + unique_sorted[1:]) / 2
            for threshold in thresholds:
                criterion_val = self._criterion_value(
                    y, feature_col, threshold)

                if criterion_val < best_gini:
                    best_gini = criterion_val
                    split_idx = feature_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        return np.array([self._traverse_tree(x, self.tree) for x in X])


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        self.n_feats = X.shape[1] if not self.n_feats else min(
            self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat_idx, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat_idx], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat_idx, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


if __name__ == '__main__':
    # Imports
    import time
    import numpy as np
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import winsound

    # option 0: hearts dataset; option 1: breast cancer dataset
    DATASET_OPTION = 0
    # option 0: my model; option 1: model by Python Engineer Youtuber;
    # option 2: sklearn model
    MODEL_OPTION = 2

    def accuracy_score(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    if DATASET_OPTION == 0:
        print("[INFO] Using UCI hearts dataset")
        df = pd.read_csv('heart.csv')
        X = df.drop(columns='target')
        y = df['target']
        X_encoded = pd.get_dummies(
            X, columns=['cp', 'restecg', 'slope', 'thal'])
        X_train, X_test, y_train, y_test = train_test_split(X_encoded,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42
                                                            )
    else:
        print("[INFO] Using breast cancer dataset")
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"{X_train.shape = }; {y_train.shape = }")
    print(f"{X_test.shape = }; {y_test.shape = }\n")

    accuracy_list = []
    time_list = []
    seeds = np.arange(0, 41)
    # seeds = np.arange(0, 2)
    for seed in seeds:
        np.random.seed(seed)
        print(f"[INFO] Using seed {seed}.")
        start_time = time.perf_counter()
        try:
            if MODEL_OPTION == 0:
                print("[INFO] Using my implementation.")
                # Accuracy = 0.8253
                clf = MyDecisionTreeClassifier(max_depth=10)
            elif MODEL_OPTION == 1:
                # Accuracy = 0.7881
                print("[INFO] Using Python Engineer's implementation.")
                clf = DecisionTree(max_depth=10)
            else:
                # Accuracy = 0.8137
                print("[INFO] Using sklearn implementation.")
                clf = DecisionTreeClassifier(criterion='gini',
                                             random_state=seed,
                                             max_depth=10)
            clf.fit(X_train, y_train)
            total_time = time.perf_counter() - start_time
            print(f"[INFO] Total training time: {total_time:.4f} secs")

            y_pred = clf.predict(X_test)
        except KeyboardInterrupt:
            raise Exception
        except:
            winsound.Beep(100, 1000)
            raise Exception
        acc = accuracy_score(y_test, y_pred)
        accuracy_list.append(acc)
        time_list.append(total_time)

        print(f"Accuracy: {acc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print()

    print(f"Avg accuracy: {np.mean(accuracy_list):.4f}")
    print(f"Avg training time: {np.mean(time_list):.4f} secs")
