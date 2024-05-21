import mlp1
import random
from utils import TRAIN, DEV, F2I, L2I, TEST, write_predictions, read_data
from collections import Counter
import numpy as np

STUDENT_1 = {"name": "Roee Esquira", "ID": "309840791"}
STUDENT_2 = {"name": "Yedidya Kfir", "ID": "209365188"}


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y = L2I_uni[label]
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
            y = L2I_uni[label]  # convert the label to number if needed.
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


def text_to_unigrams(text):
    return ["%s" % c1 for c1 in text]


TRAIN_UNI = [(l, text_to_unigrams(t)) for l, t in read_data("train")]
DEV_UNI = [(l, text_to_unigrams(t)) for l, t in read_data("dev")]
TEST_UNI = [(l, text_to_unigrams(t)) for l, t in read_data("test")]

fc_uni = Counter()
for l, feats in TRAIN_UNI:
    fc_uni.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc_uni.most_common(600)])


def feats_to_vec(features):
    identifiable_features = [
        feature_value for f in features if (feature_value := F2I_uni.get(f, None))
    ]
    feature_counter = Counter(identifiable_features)
    vec = np.zeros((1, len(F2I_uni)))
    idx = np.array(list(feature_counter.keys()))
    values = np.array(list(feature_counter.values()))
    vec[0, idx] = values
    return vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y = L2I_uni[label]
        prediction = predict(x, params)
        good += (prediction == y).sum()
        bad += (prediction != y).sum()
    return good / (good + bad)


# label strings to IDs
L2I_uni = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN_UNI]))))}
# feature strings (unigrams) to IDs
F2I_uni = {f: i for i, f in enumerate(list(sorted(vocab)))}

if __name__ == "__main__":
    # code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    bigrams = False
    if bigrams:
        params = mlp1.create_classifier(in_dim=len(F2I),
                                        hid_dim=round(len(F2I) * 1.5),
                                        out_dim=len(L2I))
        num_iter = 100
        trained_params = train_classifier(TRAIN, DEV, num_iter, 1e-3, params)
        predictions = predict_dataset(TEST, params)
        write_predictions(predictions, f"mlp1_test_num_iter{num_iter}.pred")
    else:
        params = mlp1.create_classifier(in_dim=len(F2I_uni),
                                        hid_dim=round(len(F2I_uni) * 1.5),
                                        out_dim=len(L2I_uni))
        num_iter = 50
        trained_params = train_classifier(TRAIN_UNI, DEV_UNI, num_iter, 1e-3, params)
        predictions = predict_dataset(TEST_UNI, params)
        write_predictions(predictions, f"mlp1_test_unigrams_num_iter{num_iter}.pred")
