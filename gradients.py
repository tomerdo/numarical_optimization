import numpy as np


# returns a matrix with n columns of copies of column vector vec of size m
def pump(vec, m, n, transpose=True):
    res = np.broadcast_to(vec, (n, m))
    if transpose:
        res = np.transpose(res)
    return res


# computes the softmax gradient for all weight vectors (W) together
def softmax_gradient(X, W, C):
    # adding bias row to the softmax data
    bias_row = np.ones(X.shape[1])
    X = np.vstack([X, bias_row])

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


# The next 3 functions are functions that return the transposed Jacobian of a layer
# multiplied by a vector V. Each function corresponds to a Jacobian with respect to
# a different parameter: b, W and x.
def JacV_b(relu_derivative, V):
    k = relu_derivative.shape[0]
    m = relu_derivative.shape[1]

    # iteration over l_i to compute the jacobian
    # relu_mul_v = np.zeros(k)
    RV = np.multiply(relu_derivative, V)
    return RV.sum(axis=1) / m



def JacV_w(X, relu_derivative, V):
    m = relu_derivative.shape[1]

    # computing relu derivative element wise multiply by V
    r_v = np.multiply(relu_derivative, V)
    # computing the result of this by all the samples and return the mean
    r_v_xt = np.matmul(r_v, X.transpose())
    return r_v_xt / m


def JacV_x(W, relu_derivative, V):
    m = V.shape[1]
    # computing relu derivative element wise multiply by V
    r_v = np.multiply(relu_derivative, V)

    WRV = np.matmul(W.transpose(), r_v)
    return WRV


