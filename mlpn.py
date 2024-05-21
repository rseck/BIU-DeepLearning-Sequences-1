import numpy as np

from mlp1 import glorot_init

STUDENT_1 = {"name": "Roee Esquira", "ID": "309840791"}
STUDENT_2 = {"name": "Yedidya Kfir", "ID": "209365188"}



def classifier_output(x, params):
    # YOUR CODE HERE.
    probs = []
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    return ...


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    in_dim = dims[0]
    out_dim = dims[-1]
    hidden_layers_dims = dims[1:-1]
    if len(hidden_layers_dims) == 0:
        W = glorot_init(out_dim, in_dim)
        b = glorot_init(out_dim, 1).squeeze()
        params = [W, b]
    elif len(hidden_layers_dims) == 1:
        W1 = glorot_init(hidden_layers_dims[0], in_dim)
        b1 = glorot_init(hidden_layers_dims[0], 1).squeeze()
        U = glorot_init(out_dim, hidden_layers_dims[0])
        b_tag = glorot_init(out_dim, 1).squeeze()
        params = [W1, b1, U, b_tag]
    else:
        W1 = glorot_init(hidden_layers_dims[0], in_dim)
        b1 = glorot_init(hidden_layers_dims[0], 1).squeeze()
        params = [W1, b1]
        for i in range(len(hidden_layers_dims)-1):
            W_hidden = glorot_init(hidden_layers_dims[i+1], hidden_layers_dims[i])
            b_hidden = glorot_init(hidden_layers_dims[i+1], 1).squeeze()
            params.append(W_hidden)
            params.append(b_hidden)
        w_final = glorot_init(out_dim, hidden_layers_dims[-1])
        b_final = glorot_init(out_dim, 1).squeeze()
        params.append(w_final)
        params.append(b_final)
    return params

if __name__ == "__main__":
    params = create_classifier([300,5])
    assert len(params) == 2
    params = create_classifier([300, 20, 5])
    assert len(params) == 4
    params = create_classifier([20, 30, 40, 10])
    assert len(params) == 6
    params = create_classifier([300,20,30,40,5])
    assert len(params) == 8

