#!/usr/bin/env python3

from configs import thresh

import keras.backend as K

def precision(y_true, y_pred):
    true_positives = K.sum(
        K.cast(
            K.greater(K.clip(y_true * y_pred, 0, 1), thresh), dtype="float32"),
        axis=-1)
    predicted_positives = K.sum(
        K.cast(K.greater(K.clip(y_pred, 0, 1), thresh), dtype="float32"),
        axis=-1)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(
        K.cast(
            K.greater(K.clip(y_true * y_pred, 0, 1), thresh), dtype="float32"),
        axis=-1)
    possible_positives = K.sum(
        K.cast(K.greater(K.clip(y_true, 0, 1), thresh), dtype="float32"),
        axis=-1)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), thresh), dtype=
                    "float32")) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta**2
    fbeta_score = K.mean((1 + bb) * (p * r) / (bb * p + r + K.epsilon()))
    return fbeta_score


def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred)
