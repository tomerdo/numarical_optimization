import numpy as np


# returns a matrix with n columns of copies of column vector vec of size m
def pump(vec, m, n, transpose=True):
    res = np.broadcast_to(vec, (n, m))
    if transpose:
        res = np.transpose(res)
    return res


# computes the softmax gradient for all weight vectors (W) together
def softmax_gradient(X, W, C):
    m = X.shape[1]
    l = W.shape[1]

    Xt = X.transpose()

    XtW = np.matmul(Xt, W)

    eta = XtW.max(axis=1)
    # pumping into matrix with shape (m,l)
    eta = pump(eta, m, l)

    num = np.exp(XtW - eta)
    # summing rows to get denominator
    den = num.sum(axis=1)

    # pumping into matrix with shape (m,l)
    den = pump(den, m, l)

    DexpXtw = num / den

    return (1 / m) * np.matmul(X, DexpXtw - C.transpose())


# gradient by X used for back propagation
def softmax_data_gradient(X, W, C):
    m = X.shape[1]
    l = W.shape[1]

    Wt = W.transpose()

    WtX = np.matmul(Wt, X)

    # eta = WtX.max(axis=1)
    # # pumping into matrix with shape (m,l)
    # eta = sgd.pump(eta, l, m)
    # num = np.exp(WtX - eta)

    num = np.exp(WtX)
    # summing rows to get denominator
    den = num.sum(axis=0)

    # pumping into matrix with shape (m,l)
    den = pump(den, m, l, False)

    DexpWtX = num / den

    return (1 / m) * np.matmul(W, DexpWtX - C)


# returns the gradient of the layer with respect to b as a matrix (why not as a vector?)
def ReLU_Gradient_by_b(W, X, b):
    return np.diag((np.matmul(W, X) + b) > 0)


def JacV_b(relu_derivative, V):
    k = relu_derivative.shape[0]
    m = relu_derivative.shape[1]

    # iteration over l_i to compute the jacobian
    relu_mul_v = np.zeros(k)
    for i in range(k):
        relu_mul_v[i] = np.matmul(relu_derivative[i], V[i])
    return relu_mul_v / m


def JacV_w(X, relu_derivative, V):
    k = relu_derivative.shape[0]
    m = relu_derivative.shape[1]

    v_x = np.matmul(V, X.transpose())

    relu_mul_v_x = np.zeros(k)
    for i in range(k):
        relu_mul_v_x = np.matmul(relu_derivative[i], v_x[i])
    return relu_mul_v_x / m


def JacV_x(W, relu_derivative, V):
    k = relu_derivative.shape[0]

    relu_mul_v = np.zeros(k)
    for i in range(k):
        relu_mul_v[i] = np.matmul(relu_derivative[i], V[i])

    return np.matmul(np.transpose(W), relu_mul_v)

