import numpy as np
import pandas as pd
from decision_tree_classifier import MyDecisionTreeClassifier


class MyDecisionTreeRegressor(MyDecisionTreeClassifier):
    def __init__(self, max_depth=100, min_samples_split=2, max_features=None):
        super().__init__(max_depth, min_samples_split, max_features)

    def _criterion_value(self, y, feature_col, threshold):
        """Compute the MSE as the criterion"""
        left_idxs, right_idxs = self._split(feature_col, threshold)
        y_true_left = y[left_idxs]
        y_pred_left = np.mean(y_true_left)
        y_true_right = y[right_idxs]
        y_pred_right = np.mean(y_true_right)

        n_samples = len(y_true_left) + len(y_true_right)
        mse = (1 / n_samples) * (np.sum(np.square(y_true_left - y_pred_left)) +
                                 np.sum(np.square(y_true_right - y_pred_right)))
        return mse

    def _predict_class(self, y):
        """Compute the mean as the predicted value"""
        return np.mean(y)


if __name__ == '__main__':
    # Imports
    import time
    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, r2_score
    import winsound

    # option 0: my model; option 1: sklearn model
    MODEL_OPTION = 1

    def rmse_score(y_true, y_pred):
        rmse = np.sqrt(np.sum(np.square(y_true - y_pred)))
        return rmse

    print("[INFO] Using diabetes dataset")
    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"{X_train.shape = }; {y_train.shape = }")
    print(f"{X_test.shape = }; {y_test.shape = }\n")

    rmse_list = []
    r2_list = []
    time_list = []
    seeds = np.arange(0, 41)
    # seeds = np.arange(0, 2)
    for seed in seeds:
        np.random.seed(seed)
        print(f"[INFO] Using seed {seed}.")
        start_time = time.perf_counter()
        try:
            if MODEL_OPTION == 0:
                # Avg RMSE = 625.0305
                # Avg R2 = 0.1706
                print("[INFO] Using my implementation.")
                clf = MyDecisionTreeRegressor(max_depth=20)
            else:
                # RMSE = 631.0241
                # Avg R2 = 0.1547
                print("[INFO] Using sklearn implementation.")
                clf = DecisionTreeRegressor(criterion='mse',
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
        rmse = rmse_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmse_list.append(rmse)
        r2_list.append(r2)
        time_list.append(total_time)

        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}\n")

    print(f"Avg RMSE: {np.mean(rmse_list):.4f}")
    print(f"Avg R2: {np.mean(r2_list):.4f}")
    print(f"Avg training time: {np.mean(time_list):.4f} secs")