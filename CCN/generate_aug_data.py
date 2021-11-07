import random
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import config


def contains_duplicates(X):
    indices = (X * np.array([256 ** 2, 256 ** 1, 256 ** 0])).sum(axis=1)
    # print(f'duplicates num: {abs(len(np.unique(indices)) - len(indices))}')
    return len(np.unique(indices)) != len(indices)


def filter_gray_and_white(X, y, classes):
    gray_index = classes.index('Gray')
    white_index = classes.index('White')

    # except original black and white
    mask = (y != gray_index) * (y != white_index)
    new_X = X[mask, :]
    new_y = y[mask]

    # handle gray
    gray_x = X[y == gray_index, :]
    gray_x = gray_x[gray_x.std(axis=1) == 0, :]
    new_X = np.append(new_X, gray_x, axis=0)
    new_y = np.append(new_y, np.ones(gray_x.shape[0]) * gray_index)

    # handle white
    white_x = X[y == white_index, :]
    white_x = white_x[white_x.std(axis=1) == 0, :]
    new_X = np.append(new_X, white_x, axis=0)
    new_y = np.append(new_y, np.ones(white_x.shape[0]) * white_index)

    # assert not contains_duplicates(new_X)
    return new_X, new_y


def augment_black_gray_and_white(X, y, classes):
    """
    Black: [0, 0, 0] - [43, 43, 43]
    Gray: [44, 44, 44] - [244, 244, 244]
    White: [245, 245, 245] - [255, 255, 255]
    """
    black_index = classes.index('Black')
    gray_index = classes.index('Gray')
    white_index = classes.index('White')

    new_X = np.copy(X)
    new_y = np.copy(y)
    t = np.ones_like(y)

    # original black, gray and white
    o_black_x = X[y == black_index, :]
    o_gray_x = X[y == gray_index, :]
    o_white_x = X[y == white_index, :]

    # handle black
    l_black_x = o_black_x[o_black_x.std(axis=1) == 0, :]
    l_black_idx = l_black_x.mean(axis=1).tolist()
    n_black_idx = list(set([i for i in range(0, 43 + 1)]) - set(l_black_idx))
    n_black_x = np.array(n_black_idx).reshape(-1, 1).repeat(3, 1)
    new_X = np.append(new_X, n_black_x, axis=0)
    new_y = np.append(new_y, np.ones(n_black_x.shape[0]) * black_index)
    black_t = np.zeros(n_black_x.shape[0])
    t = np.append(t, black_t)

    # handle gray
    l_gray_x = o_gray_x[o_gray_x.std(axis=1) == 0, :]
    l_gray_idx = l_gray_x.mean(axis=1).tolist()
    n_gray_idx = list(set([i for i in range(44, 244 + 1)]) - set(l_gray_idx))
    n_gray_x = np.array(n_gray_idx).reshape(-1, 1).repeat(3, 1)
    new_X = np.append(new_X, n_gray_x, axis=0)
    new_y = np.append(new_y, np.ones(n_gray_x.shape[0]) * gray_index)
    gray_t = np.zeros(n_gray_x.shape[0])
    t = np.append(t, gray_t)

    # handle white
    l_white_x = o_white_x[o_white_x.std(axis=1) == 0, :]
    o_white_idx = l_white_x.mean(axis=1).tolist()
    n_white_idx = list(set([i for i in range(245, 255 + 1)]) - set(o_white_idx))
    n_white_x = np.array(n_white_idx).reshape(-1, 1).repeat(3, 1)
    new_X = np.append(new_X, n_white_x, axis=0)
    new_y = np.append(new_y, np.ones(n_white_x.shape[0]) * white_index)
    white_t = np.zeros(n_white_x.shape[0])
    t = np.append(t, white_t)

    return new_X, new_y, t


def augment_gray_and_white(X, y, classes):
    """
    Black: [0, 0, 0] - [43, 43, 43]
    Gray: [44, 44, 44] - [244, 244, 244]
    White: [245, 245, 245] - [255, 255, 255]
    """

    gray_index = classes.index('Gray')
    white_index = classes.index('White')

    # except original black and white
    mask = (y != gray_index) * (y != white_index)
    new_X = X[mask, :]
    new_y = y[mask]
    t = np.ones_like(new_y)

    # original gray and white
    # o_gray_x = X[y == gray_index, :]
    o_gray_x_indices = (X[y == gray_index, :] * np.array([256 ** 2, 256 ** 1, 256 ** 0])).sum(axis=1)
    # o_white_x = X[y == white_index, :]
    o_white_x_indices = (X[y == white_index, :] * np.array([256 ** 2, 256 ** 1, 256 ** 0])).sum(axis=1)

    # handle gray
    gray_x = np.array([i for i in range(44, 244 + 1)]).reshape(-1, 1).repeat(3, axis=1)
    new_X = np.append(new_X, gray_x, axis=0)
    new_y = np.append(new_y, np.ones(gray_x.shape[0]) * gray_index)
    # intersect1d
    gray_x_indices = (gray_x * np.array([256 ** 2, 256 ** 1, 256 ** 0])).sum(axis=1)
    _, indices, _, = np.intersect1d(gray_x_indices, o_gray_x_indices, return_indices=True)
    gray_t = np.zeros(gray_x.shape[0])
    gray_t[indices] = 1
    # print(indices)
    # print((gray_t == 1).sum().item())
    t = np.append(t, gray_t)

    # handle white
    white_x = np.array([i for i in range(245, 255 + 1)]).reshape(-1, 1).repeat(3, axis=1)
    new_X = np.append(new_X, white_x, axis=0)
    new_y = np.append(new_y, np.ones(white_x.shape[0]) * white_index)
    # intersect1d
    white_x_indices = (white_x * np.array([256 ** 2, 256 ** 1, 256 ** 0])).sum(axis=1)
    # print(white_x_indices)
    # print(o_white_x_indices)
    _, indices, _, = np.intersect1d(white_x_indices, o_white_x_indices, return_indices=True)
    # print(indices)
    white_t = np.zeros(white_x.shape[0])
    white_t[indices] = 1
    # print((white_t == 1).sum().item())
    t = np.append(t, white_t)
    # assert not contains_duplicates(new_X)
    return new_X, new_y, t


def weighted_average(X_indices, xs_x, other_x, num, class_index):
    counter = num
    # X_indices = (X * np.array([256 ** 2, 256 ** 1, 256 ** 0])).sum(axis=1)
    synthetic_x = []
    while counter > 0:
        xs = xs_x[random.randint(0, xs_x.shape[0] - 1)]
        xo = other_x[random.randint(0, other_x.shape[0] - 1)]
        new_x = np.random.rand(1) * (xs - xo) + xo

        new_x = new_x.round()  # round
        new_x[new_x > 255.] = 255.  # boundary check
        # duplication check
        index = (new_x * np.array([256 ** 2, 256 ** 1, 256 ** 0])).sum()

        not_duplicated = (X_indices == index).sum().item() == 0

        if not_duplicated:
            # indices.append(index)
            X_indices = np.append(X_indices, index)
            synthetic_x.append(new_x)
            counter -= 1

    assert len(synthetic_x) == num
    # assert not contains_duplicates(np.array(synthetic_x))
    return np.array(synthetic_x), np.ones(len(synthetic_x)) * class_index, np.zeros(len(synthetic_x))


def generate_data(df, classes: List, augment_classes: List, num=500):
    # df to array
    X = df.iloc[:, :3].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df.Group)
    assert encoder.classes_.tolist() == classes
    assert not contains_duplicates(X)
    assert X.shape[0] == y.shape[0]

    # filter gray and white
    X, y = filter_gray_and_white(X, y, classes)
    assert not contains_duplicates(X)
    assert X.shape[0] == y.shape[0]
    print(X.shape, y.shape)

    # augment gray and white
    # X, y, t = augment_gray_and_white(X, y, classes)
    X, y, t = augment_black_gray_and_white(X, y, classes)
    assert not contains_duplicates(X)
    assert X.shape[0] == y.shape[0] == t.shape[0]
    print(X.shape, y.shape, t.shape)

    # augment selected classes
    xs_data = df[df['Source'] == 'xkcd-satfaces']
    other_data = df[df['Source'] != 'xkcd-satfaces']

    for k in augment_classes:
        X_indices = (X * np.array([256 ** 2, 256 ** 1, 256 ** 0])).sum(axis=1)
        class_index = classes.index(k)
        xs_x = xs_data[xs_data['Group'] == k].iloc[:, :3].values
        other_x = other_data[other_data['Group'] == k].iloc[:, :3].values
        s_x, s_y, s_t = weighted_average(X_indices, xs_x, other_x, num, class_index)
        X = np.append(X, s_x, axis=0)
        y = np.append(y, s_y)
        t = np.append(t, s_t)
        print(f'{k}, {X.shape}, {y.shape}, {t.shape}')
        assert not contains_duplicates(X)

    assert not contains_duplicates(X)

    return X, y, t


if __name__ == '__main__':
    xlsx_file_path = '/root/data/dataset.xlsx'
    sheet_name = 'All'
    df = pd.read_excel(xlsx_file_path, sheet_name)
    # sources = ['wikipedia', 'fs595c1', 'hollasch', 'xkcd-satfaces']
    df = pd.DataFrame(df[df['Source'] != 'rgb'])  # except source: 'rgb'
    print(f'df.shape: {df.shape}')

    augment_classes = list(set(config['classes']) - {'Black', 'Gray', 'White'})
    num = 400

    X, y, t = generate_data(df, config['classes'], augment_classes, num)
    data = np.hstack((X, np.expand_dims(y, 1), np.expand_dims(t, 1)))  # [m, 3+1+1]
    assert (y != data[:, 3]).sum().item() == 0
    assert (t != data[:, 4]).sum().item() == 0
    print(f'total synthetic num: {(t == 0).sum().item()}')
    # np.save(f'./data/aug_data.npy', data)
    # print('saved!')
