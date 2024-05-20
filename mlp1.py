import numpy as np

from loglinear import softmax

STUDENT = {"name": "YOUR NAME", "ID": "YOUR ID NUMBER"}


def classifier_output(x, params):
    W, b, U, b_tag = params
    first_layer_output = (W @ x) + b
    tanh_res = np.tanh(first_layer_output).squeeze()
    second_layer_output = (U.T @ tanh_res.T).T + b_tag.reshape(1, len(b_tag))
    return softmax(second_layer_output)


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    probs = classifier_output(x, params)
    loss = -np.log(probs[..., y]).sum()
    y_vec = np.zeros_like(probs, dtype=int)
    y_vec[..., y] = 1
    softmax_gradient = probs - y_vec

    first_layer_output = (W.T @ x.T).T + b.reshape(1, len(b))
    tanh_res = np.tanh(first_layer_output).squeeze()
    gU = (tanh_res.T @ softmax_gradient) / tanh_res.shape[0]
    gb_tag = softmax_gradient.mean(axis=0)


    return ...


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = glorot_init(hid_dim, in_dim)
    b = glorot_init(hid_dim, 1).squeeze()
    U = glorot_init(out_dim, in_dim)
    b_tag = glorot_init(out_dim, 1).squeeze()
    return [W, b, U, b_tag]


def glorot_init(first_dim, second_dim):
    return np.ones((first_dim, second_dim))
    epsilon = np.sqrt(6 / (first_dim + second_dim))
    return np.random.uniform(-epsilon, epsilon, (first_dim, second_dim))


if __name__ == "__main__":
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    in_dim, hid_dim, out_dim = (3, 4, 5)
    W, b, U, b_tag = create_classifier(in_dim, hid_dim, out_dim)
    x = np.ones(in_dim)
    probs = classifier_output(x, [W, b, U, b_tag])

    loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])


    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[1]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
