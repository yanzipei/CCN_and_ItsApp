import argparse
import os
import warnings
from datetime import datetime

import numpy as np
from joblib import dump
from sklearn import svm
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import config
from util import save_yaml_file

warnings.filterwarnings("ignore")


def main(args):
    log_dir = args.log_dir if args.log_dir[-1] == '/' else args.log_dir + '/'
    log_dir += datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    save_yaml_file(log_dir, args, 'config.yml')

    data = np.load(args.log_data)
    X = data[:, 0:3]
    y = data[:, 3]
    # t = data[:, 4]
    # classes = config['classes']

    # print(f'X.shape: {X.shape}, y.shape: {y.shape}')
    # print(f'target classes: {classes}')

    # normalize data
    mean = np.array(config['mean'])
    std = np.array(config['std'])

    # print(f'before normalization: mean={X.mean()}, std={X.std()}')
    X = (X - mean) / std
    # print(f'after normalization: mean={X.mean()}, std={X.std()}')

    # num_classes = len(classes)
    # num_features = 3

    # shuffle = False
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=False)

    score_y = []
    pred_y = np.array([])
    true_y = np.array([])
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        current_log_dir = log_dir + f'/fold_{i}'
        if not os.path.exists(current_log_dir):
            os.makedirs(current_log_dir)

        print(f'fold {i}:')
        train_X, train_y, test_X, test_y = X[train_indices], y[train_indices], X[test_indices], y[test_indices]

        true_y = np.append(true_y, test_y)

        clf = svm.SVC(C=args.C, probability=True)
        clf = clf.fit(train_X, train_y)
        # save
        dump(clf, f'{current_log_dir}/clf.joblib')
        # from joblib import dump, load
        # clf = load('clf.joblib')
        # predict
        pred_y = np.append(pred_y, clf.predict(test_X))
        score_y.append(clf.predict_proba(test_X))

    # to np.ndarray
    score_y = np.concatenate(score_y, axis=0)
    # confusion matrix
    cm = confusion_matrix(y_true=true_y, y_pred=pred_y)
    np.save(f'{log_dir}/cm.npy', cm)

    # save config
    result_dict = {}

    top1_acc = top_k_accuracy_score(true_y, score_y, k=1)
    top5_acc = top_k_accuracy_score(true_y, score_y, k=5)
    result_dict['top1_acc'], result_dict['top5_acc'] = top1_acc, top5_acc
    save_yaml_file(log_dir, result_dict, 'result.yml')

    print(result_dict)


if __name__ == '__main__':
    """
    python tune_svm.py --data ./data/data.npy --log_dir ./runs/tune/svm
    python tune_svm.py --data ./data/data_except_bgw_400.npy --log_dir ./runs/tune/svm
    """

    parser = argparse.ArgumentParser(description='tuning SVM for Computational Color Naming.')

    parser.add_argument('--C', default=8000.0, type=float,
                        help='Regularization parameter. '
                             'The strength of the regularization is inversely proportional to C. '
                             'Must be strictly positive. The penalty is a squared l2 penalty.')

    parser.add_argument('--data', default='./data/aug_data.npy',
                        help='the training data, ./data/data.npy or ./data/aug_data.npy,\n'
                             'default: ./data/aug_data.npy')

    parser.add_argument('--num_folds', default=5, type=int,
                        help='numbers of folds for cross-validation, default: 5')

    parser.add_argument('--log_dir', default='./runs/tune/svm', type=str,
                        help='the directory where checkpoint and logs to be saved, default: ./runs/tune/svm')

    args = parser.parse_args()
    main(args)
