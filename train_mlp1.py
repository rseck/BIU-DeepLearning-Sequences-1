import mlp1
import random
from collections import Counter
import numpy as np

STUDENT_1 = {"name": "Roee Esquira", "ID": "309840791"}
STUDENT_2 = {"name": "Yedidya Kfir", "ID": "209365188"}

# utils segment
def read_data(fname):
    data = []
    for line in open(fname, "rb"):
        label, text = line.strip().lower().decode().split("\t", 1)
        data.append((label, text))
    return data


def write_predictions(predictions, fname):
    I2L = {i: l for l, i in L2I.items()}
    with open(fname, "w") as f:
        for pred in predictions:
            f.write(str(I2L[pred]) + "\n")


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data("train")]
DEV = [(l, text_to_bigrams(t)) for l, t in read_data("dev")]
TEST = [(l, text_to_bigrams(t)) for l, t in read_data("test")]

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}
# end utils segment


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
