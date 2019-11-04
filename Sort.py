import numpy as np


def sort_for_plotting(X, y):
    data_set = np.column_stack((X, y))
    test_set_sorted = data_set[data_set[:, 0].argsort()]
    X_sorted = test_set_sorted[:, 0]
    y_sorted = test_set_sorted[:, 1]

    return X_sorted, y_sorted
