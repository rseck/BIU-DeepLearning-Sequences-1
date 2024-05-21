import mlp1
import random
from train_loglin import feats_to_vec
from utils import TRAIN, DEV, F2I, L2I, TEST, write_predictions
from collections import Counter
import numpy as np

STUDENT_1 = {"name": "Roee Esquira", "ID": "309840791"}
STUDENT_2 = {"name": "Yedidya Kfir", "ID": "209365188"}


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
    W, b, U, b_tag = params
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = L2I[label]  # convert the label to number if needed.
            loss, (gW, gb, gU, gb_tag) = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            W -= learning_rate * gW
            b -= learning_rate * gb
            U -= learning_rate * gU
            b_tag -= learning_rate * gb_tag

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def predict(x, params):
    x = x.squeeze()
    return np.argmax(mlp1.classifier_output(x, params))


def predict_dataset(dataset, params):
    return [predict(feats_to_vec(features), params) for _, features in dataset]


if __name__ == "__main__":
    # code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    params = mlp1.create_classifier(in_dim=len(F2I),
                                    hid_dim=round(len(F2I) * 1.5),
                                    out_dim=len(L2I))
    num_iter = 100
    trained_params = train_classifier(TRAIN, DEV, num_iter, 1e-3, params)
    predictions = predict_dataset(TEST, params)
    write_predictions(predictions, f"mlp1_test_num_iter{num_iter}.pred")
