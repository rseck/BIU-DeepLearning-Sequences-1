import numpy as np
from loglinear import softmax
from mlp1 import glorot_init

STUDENT_1 = {"name": "Roee Esquira", "ID": "309840791"}
STUDENT_2 = {"name": "Yedidya Kfir", "ID": "209365188"}


def classifier_output(x, params):
    if len(params) == 2:
        W, b = params
        return softmax(np.dot(x, W) + b)
    elif len(params) == 4:
        W_1, b_1, W_2, b_2 = params
        first_layer_output = np.dot(x, W_1) + b_1
        tanh_res = np.tanh(first_layer_output)
        second_layer_output = np.dot(tanh_res, W_2) + b_2
        return softmax(second_layer_output)
    else:
        grouped_params = [(params[i], params[i + 1]) for i in range(0, len(params), 2)]
        tanh_res = x
        for i in range(len(grouped_params) - 1):
            W_i, b_i = grouped_params[i]
            i_layer_output = np.dot(tanh_res, W_i) + b_i
            tanh_res = np.tanh(i_layer_output)
        W_n, b_n = grouped_params[-1]
        last_layer_output = np.dot(tanh_res, W_n) + b_n
        return softmax(last_layer_output)


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
    x = np.array(x)
    probs = classifier_output(x, params).squeeze()
    loss = -np.log(probs)
    y_vec = np.zeros_like(probs, dtype=int)
    y_vec[y] = 1
    softmax_gradient = probs - y_vec

    if len(params) == 2:
        gW = np.outer(x, softmax_gradient)
        gb = softmax_gradient
        return loss, [gW, gb]
    elif len(params) == 4:
        W_1, b_1, W_2, b_2 = params
        tanh_result = np.tanh(np.dot(x, W_1) + b_1)
        gW_2 = np.outer(tanh_result, softmax_gradient)
        gb_2 = softmax_gradient
        gb_1 = np.dot(softmax_gradient, W_2) * (1 - (tanh_result * tanh_result))
        gW_1 = np.outer(x, gb_1)
        return loss, [gW_1, gb_1, gW_2, gb_2]
    else:
        grouped_params = [(params[i], params[i + 1]) for i in range(0, len(params), 2)]
        tanh_results = [x]
        layer_outputs = [x]
        for W_i, b_i in grouped_params:
            i_layer_output = np.dot(layer_outputs[-1], W_i) + b_i
            layer_outputs.append(i_layer_output)
            tanh_res = np.tanh(i_layer_output)
            tanh_results.append(tanh_res)

        gWn = np.outer(tanh_results[-2], softmax_gradient)
        gbn = softmax_gradient
        grads = [gWn, gbn]

        for i in range(len(grouped_params) - 2, -1, -1):
            gb_i = (np.dot(grouped_params[i+1][0], grads[1])) * (1-tanh_results[i+1] * tanh_results[i+1])
            gw_i = np.outer(layer_outputs[i], gb_i)
            grads.insert(0, gb_i.squeeze())
            grads.insert(0, gw_i)
        return loss, grads


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
        W = glorot_init(in_dim, out_dim)
        b = glorot_init(out_dim, 1)
        params = [W, b]
    elif len(hidden_layers_dims) == 1:
        W1 = glorot_init(hidden_layers_dims[0], in_dim)
        b1 = glorot_init(hidden_layers_dims[0], 1)
        U = glorot_init(hidden_layers_dims[0], out_dim)
        b_tag = glorot_init(out_dim, 1)
        params = [W1, b1, U, b_tag]
    else:
        W1 = glorot_init(in_dim, hidden_layers_dims[0])
        b1 = glorot_init(hidden_layers_dims[0], 1)
        params = [W1, b1]
        for i in range(len(hidden_layers_dims) - 1):
            W_hidden = glorot_init(hidden_layers_dims[i], hidden_layers_dims[i + 1])
            b_hidden = glorot_init(hidden_layers_dims[i + 1], 1)
            params.append(W_hidden)
            params.append(b_hidden)
        w_final = glorot_init(hidden_layers_dims[-1], out_dim)
        b_final = glorot_init(out_dim, 1)
        params.append(w_final)
        params.append(b_final)
    return params


if __name__ == "__main__":
    params = create_classifier([20, 5])
    assert len(params) == 2
    x = np.random.randn(20)
    probs = classifier_output(x, params)
    assert probs.shape == (5,)
    loss_and_gradients(np.ones(20), 0, params)

    params = create_classifier([300, 20, 5])
    assert len(params) == 4
    x = np.ones(300)
    probs = classifier_output(x, params)
    assert probs.shape == (5,)
    loss_and_gradients(np.ones(300), 0, params)

    params = create_classifier([20, 30, 40, 10])
    assert len(params) == 6
    x = np.ones(20)
    probs = classifier_output(x, params)
    assert probs.shape == (10,)
    loss_and_gradients(np.ones(20), 0, params)

    params = create_classifier([300, 20, 30, 40, 5])
    assert len(params) == 8
    x = np.ones(300)
    probs = classifier_output(x, params)
    assert probs.shape == (5,)
    loss_and_gradients(np.ones(300), 0, params)
