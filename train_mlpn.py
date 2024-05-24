import random
from datetime import datetime

import mlpn
from utils import TRAIN, DEV, F2I, L2I, TEST, write_predictions
import numpy as np
from collections import Counter


STUDENT_1 = {"name": "Roee Esquira", "ID": "309840791"}
STUDENT_2 = {"name": "Yedidya Kfir", "ID": "209365188"}


def feats_to_vec(features):
    identifiable_features = [
        feature_value for f in features if (feature_value := F2I.get(f, None))
    ]
    feature_counter = Counter(identifiable_features)
    vec = np.zeros((1, len(F2I)))
    idx = np.array(list(feature_counter.keys()))
    values = np.array(list(feature_counter.values()))
    vec[0, idx] = values
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


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = L2I[label]  # convert the label to number if needed.
            loss, grads = mlpn.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            for param, grad in zip(params, grads):
                param -= learning_rate * grad

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def predict(x, params):
    x = x.squeeze()
    return np.argmax(mlpn.classifier_output(x, params))


def predict_dataset(dataset, params):
    return [predict(feats_to_vec(features), params) for _, features in dataset]


if __name__ == "__main__":
    # code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    # todo remove seed fix before submit
    # seed 42 capped in 0.853 after 4 iterations
    # np.random.seed(0)
    for i in range(5):
        params = mlpn.create_classifier([len(F2I),
                                         len(F2I) * 2,
                                         len(F2I),
                                         round(len(F2I) / 2),
                                         len(L2I)])
        num_iter = 20
        trained_params = train_classifier(TRAIN, DEV, num_iter, 1e-3, params)
        predictions = predict_dataset(TEST, params)
        write_predictions(predictions, f"{datetime.now()}_exp_{i}_mlpn_with_dims_{str([len(F2I), len(F2I) * 2, len(F2I) * 2, len(F2I),len(L2I)])}_test_num_iter{num_iter}.pred")
