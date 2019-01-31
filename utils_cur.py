# Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
import numpy as np

def eval_classifier(classifier_obj, X, y):
    num_correct = 0
    for i, x_test in enumerate(X):
        y_pred = classifier_obj.predict(x_test.reshape(1, -1))
        y_gt = y[i]
        print('y_gt = {}, y_pred = {}'.format(y_gt, y_pred))
        if y_gt == y_pred:
            num_correct += 1
    print('Accuracy =', num_correct * 100 / len(X))

def predict_test_case(classifier_obj, i, X, y, predict_prob=False):
    y_pred = classifier_obj.predict(X[i].reshape(1, -1))
    y_gt = y[i]
    print('Input feature vector:')
    for feature_val in X[i]:
        print(feature_val, end=', ')
    print()
    print('C code:')
    for j, feature_val in enumerate(X[i]):
        print('input_fvec[{}] = {};'.format(j, feature_val))
    print('y_gt = {}, y_pred = {}'.format(y_gt, y_pred))
    if predict_prob:
        probs_pred = classifier_obj.predict_proba(X[i].reshape(1, -1))[0]
        print('probs_pred:', end=' ')
        for i in range(len(probs_pred)):
            print('{:f}'.format(probs_pred[i]), end=', ')
        print()

def softmax_v1(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

def softmax_v2(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def softmax_v3(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p