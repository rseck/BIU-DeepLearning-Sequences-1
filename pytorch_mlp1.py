import torch
import torch.nn as nn
import torch.optim as optim


def accuracy_on_dataset(dataset, params):
    model, opt, loss = params
    good = bad = 0.0
    for label, features in dataset:
        # x = feats_to_vec(features)
        # y = L2I_uni[label]
        x = torch.tensor([features], dtype=torch.float)
        y = label
        prediction = model(x).argmax()
        good += (prediction == y).sum()
        bad += (prediction != y).sum()
    return good / (good + bad)



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    model, opt, loss_func = params
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        # random.shuffle(train_data)
        for label, features in train_data:
            # x = feats_to_vec(features)  # convert features to a vector.
            # y = L2I_uni[label]  # convert the label to number if needed.
            x = torch.tensor([features], dtype=torch.float)
            y = torch.tensor(label, dtype=torch.float)
            model.zero_grad()
            output = model(x)[0]
            y = torch.zeros_like(output)
            y[label] = 1.0
            loss = loss_func(output, y)
            # loss, (gW, gb, gU, gb_tag) = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            loss.backward()
            opt.step()


        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


# Hyperparameters
input_size = 2  # Example for MNIST dataset (28x28 pixels)
hidden_size = 3
num_classes = 2
learning_rate = 1e-3

# Create the model
model = MLP(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

xor_data = [(0, [0, 0]), (0, [0, 1]), (0, [1, 0]), (1, [1, 1])]
num_iter = 1000
trained_params = train_classifier(xor_data, xor_data, num_iter, 1e-3, (model, optimizer, criterion))
# predictions = predict_dataset(xor_data, params)
# write_predictions(predictions, f"xor_mlp1_num_iter{num_iter}.pred")

# Example input tensor
x = torch.randn(64, input_size)  # Batch size of 64

# Forward pass (example)
outputs = model(x)
# print(outputs)
