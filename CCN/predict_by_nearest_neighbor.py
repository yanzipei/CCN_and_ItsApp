import itertools

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def load_X():
    X = []
    for (i, j, k) in itertools.product(range(256), range(256), range(256)):
        X.append([i, j, k])
    X = np.asarray(X).astype(np.float)
    # print(X.size())
    return X


if __name__ == '__main__':
    data = np.load('./data_backup/cleaned_data.npy')
    X, y = data[:, 0:3], data[:, 3]
    print(X.shape, y.shape)
    clf = KNeighborsClassifier(n_neighbors=1, weights='distance').fit(X, y)

    all_X = load_X()
    print(all_X.shape)
    pred_y = clf.predict(all_X)
    print(pred_y.shape)
    np.save('./data_backup/y_nearest_neighbor', pred_y)
