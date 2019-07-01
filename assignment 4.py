import scipy.io
import numpy as np
from matplotlib import pyplot
from scipy.linalg import norm


# returns a matrix with n columns of copies of column vector vec of size m
def pump(vec, m, n, transpose=True):
    res = np.broadcast_to(vec, (n, m))
    if transpose:
        res = np.transpose(res)
    return res


def softmax_objective(X, W, C, b=0):
    n = X.shape[0]
    m = X.shape[1]
    l = W.shape[1]

    Xt = X.transpose()
    XtW = np.matmul(Xt, W)

    if b != 0:
        # pumping into matrix with shape (m,l)
        B = pump(b, l, m, False)
    else:
        B = 0

    XtW = XtW - B
    eta = XtW.max(axis=1)

    # pumping into matrix with shape (m,l)
    eta = pump(eta, m, l)

    num = np.exp(XtW - eta)

    # summing rows to get denominator
    den = num.sum(axis=1)

    # pumping into matrix with shape (m,l)
    den = pump(den, m, l)

    logDXtW = np.log(num / den)

    res = 0

    for i in range(l):
        res += (-1/m)*sum(C[i, :]*logDXtW[:, i])

    return res


# computes the softmax gradient for all weight vectors together
def softmax_gradient(X, W, C, b=0):
    m = X.shape[1]
    Xt = X.transpose()

    XtW = np.matmul(Xt,W)
    expXtW = np.exp(XtW)
    D = np.diag(1/(expXtW.sum(axis=1)))
    DexpXtw = np.matmul(D, expXtW)

    return (1/m)*np.matmul(X, DexpXtw-C.transpose())


def func(X, W, C, w):
    W1 = np.zeros(W.shape)
    W1 += W
    W1[:, 0] = w
    return softmax_objective(X, W1, C)


def gradient(X, W, C, w):
    W1 = np.zeros(W.shape)
    W1 += W
    W1[:, 0] = w
    return softmax_gradient(X, W, C)






if __name__ == "__main__":

    # A = np.asarray([[1,1,4],[1,1,1]])
    # W = np.asarray([[0,0,1,0,0,0],[0,0,0,0,0,1]])
    # C = np.asarray([[0,0,0],[0,0,0],[1,0,0],[0,0,1],[0,0,0],[0,1,0]])
    #
    # print(softmax_objective(A, W, C))

    X = np.random.rand(3, 5)
    W = np.random.rand(3, 3)
    c = np.random.randint(0, 3, 5)
    C = np.asarray([c == 0, c == 1, c == 2])
    w = W[:, 0]

    d = np.random.rand(3)
    epsilon = 1

    f = lambda w: func(X, W, C, w)
    gradf = lambda w: gradient(X, W, C, w)

    curr = f(w)
    last = f(w)

    for i in range(10):
        last = curr
        epsilon = epsilon*0.5
        curr = abs(f(w+epsilon*d)-f(w))
        print('ratio 1 is: ', curr/last)

    d = np.random.rand(3)
    epsilon = 0.2

    curr = f(w)
    last = f(w)

    for i in range(10):
        last = curr
        epsilon = epsilon * 0.5
        curr = abs(f(w + epsilon * d) - f(w)-epsilon*sum(d*(gradf(w)[:, 0])))
        print('ratio 2 is: ', curr / last)


    # GMMData = scipy.io.loadmat('GMMData.mat')
    # PeaksData = scipy.io.loadmat('PeaksData.mat')
    # SwissRollData = scipy.io.loadmat('SwissRollData.mat')
    #
    # SwissCt = SwissRollData['Ct']
    #
    # print(SwissCt.shape)
    #
    # PeaksCt = PeaksData['Yt']
    #
    # print(PeaksCt.shape)
    #
    # print(PeaksCt[0].shape)
    #
    # print(type(GMMData['Cv']))
    # print(type(GMMData['Ct']))
    # print(type(GMMData['Yv']))
    # print(type(GMMData['Yt']))






