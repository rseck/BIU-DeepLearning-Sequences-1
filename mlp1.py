import numpy as np
from loglinear import softmax

STUDENT_1 = {"name": "Roee Esquira", "ID": "309840791"}
STUDENT_2 = {"name": "Yedidya Kfir", "ID": "209365188"}



def classifier_output(x, params):
    W, b, U, b_tag = params
    first_layer_output = np.dot(x, W) + b
    tanh_res = np.tanh(first_layer_output)
    second_layer_output = np.dot(tanh_res, U) + b_tag
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
    x = np.array(x).squeeze()
    probs = classifier_output(x, params)
    # print(probs.sum())
    loss = -np.log(probs[y])
    y_vec = np.zeros_like(probs, dtype=int)
    y_vec[y] = 1
    softmax_gradient = probs - y_vec

    tanh_result = np.tanh(np.dot(x, W) + b)
    gU = np.outer(tanh_result, softmax_gradient)
    gb_tag = softmax_gradient
    gb = np.dot(U, softmax_gradient) * (1 - (tanh_result * tanh_result))
    gW = np.outer(x, gb)

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = glorot_init(in_dim, hid_dim)
    b = glorot_init(hid_dim, 1)
    U = glorot_init(hid_dim, out_dim)
    b_tag = glorot_init(out_dim, 1)
    return [W, b, U, b_tag]


def glorot_init(first_dim, second_dim):
    epsilon = np.sqrt(6 / (first_dim + second_dim))
    return np.random.uniform(-epsilon, epsilon, first_dim) if second_dim == 1 else np.random.uniform(-epsilon, epsilon, (first_dim, second_dim))


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
        global U
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        global U
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[1]


    def _loss_and_U_grad(U):
        global b
        global W
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        global W
        global U
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)
