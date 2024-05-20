import random

import numpy as np

import loglinear as ll
from utils import TRAIN, DEV, F2I, L2I, TEST, write_predictions

STUDENT = {"name": "Yedidya Kfir", "ID": "209365188"}


def feats_to_vec(features):
    identifiable_features = [
        feature_value for f in features if (feature_value := F2I.get(f, None))
    ]
    real_features_size = len(identifiable_features)
    vec = np.zeros((real_features_size, len(F2I)))
    idx = np.array(identifiable_features)
    vec[np.arange(real_features_size), idx] = 1
    return vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y = L2I[label]
        prediction = predict(x, params)
        good += (prediction == y).sum()
        bad += (prediction != y).sum()
    return good / (good + bad)


def most_frequent(arr):
    counts = np.bincount(arr)
    return np.argmax(counts)


def predict_dataset(dataset, params):
    return [most_frequent(predict(feats_to_vec(features), params)) for _, features in dataset]


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    w, b = params
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = L2I[label]  # convert the label to number if needed.
            loss, (w_grad, b_grad) = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            w -= learning_rate * w_grad
            b -= learning_rate * b_grad
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def predict(x, params):
    return np.argmax(ll.classifier_output(x, params), axis=1)


if __name__ == "__main__":
    params = ll.create_classifier(len(F2I), len(L2I))
    trained_params = train_classifier(TRAIN, DEV, 1000, 1e-3, params)
    predictions = predict_dataset(TEST, params)
    write_predictions(predictions, "test.pred")
