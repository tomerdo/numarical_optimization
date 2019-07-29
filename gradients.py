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

    # # adding bias row to the softmax data
    bias_row = np.ones(X.shape[1])
    X = np.vstack([X, bias_row])


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
    # m = relu_derivative.shape[1]
    #
    # r_v = np.multiply(relu_derivative, V)
    #
    # return (r_v.sum(axis=1) / m).reshape(r_v.shape)

    return np.multiply(relu_derivative, V)



def JacV_w(X, der, V):
    m = der.shape[1]
    k = V.shape[0]
    n = X.shape[0]

    # this works for some reason for m = 1, k = 1
    # computing relu derivative element wise multiply by V
    r_v = np.multiply(der, V)
    # computing the result of this by all the samples and return the mean
    r_v_xt = np.matmul(r_v.transpose(), X)

    return r_v_xt

    # res = np.zeros((k, m))
    #
    # for i in range(m):
    #
    #     V[:, i] = np.multiply(der[:, i], V[:, i])
    #
    #     v_xt = np.matmul(V[:, i].reshape(k, 1), X[:, i].reshape(1, n))
    #
    #     # der_v_xt = np.multiply(der[:, i].reshape(k, 1), v_xt)
    #     # res += der_v_xt
    #
    #     res += v_xt
    #
    # return res/m

    # # does not work at all
    # res = np.zeros((k, n))
    #
    # for i in range(m):
    #     xt_kronI = np.kron(np.eye(n), X[:, i])
    #
    #     der_xt_kronI = np.multiply(relu_derivative[:, i].reshape(k, 1), xt_kronI)
    #
    #     res += np.matmul(der_xt_kronI.transpose(), V[:, i].reshape(k, 1)).reshape(n, k).transpose()
    #
    # return res



    # res = np.zeros((n, m))
    #
    # for i in range(m):
    #     res[:, i] = np.matmul(np.multiply(relu_derivative[:, i].reshape(k, 1), W.transpose()), V[:, i])
    #
    # return res


    # ======================================================================================
    # ======================================================================================
    # ======================================================================================

    # # this is what I calculated that should work
    #
    # der_x = np.matmul(relu_derivative,X.transpose())
    #
    # der_x_v = np.matmul(der_x, V)
    #
    # return der_x_v / m

    # ======================================================================================
    # ======================================================================================
    # ======================================================================================


def JacV_x(W, relu_derivative, V):
    m = V.shape[1]
    k = V.shape[0]
    n = W.shape[0]

    # this works, but does not make sense.
    der = relu_derivative.flatten('F').reshape(k * m, 1)

    # v = V.reshape(v_shape[0] * v_shape[1], 1)
    v = V.flatten('F').reshape(k * m, 1)

    Wkron = np.kron(np.eye(m), W.transpose())

    derW = np.multiply(der, Wkron)

    derWV = np.matmul(derW, v)

    return derWV.reshape(m, k).transpose()


    # # THIS WORKS, BUT IS STILL ILLOGICAL!!!!!!
    # res = np.zeros((n, m))
    #
    # for i in range(m):
    #     res[:, i] = np.matmul(np.multiply(relu_derivative[:, i].reshape(k, 1), W.transpose()), V[:, i])
    #
    # return res





