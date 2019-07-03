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
    # n = X.shape[0]
    m = X.shape[1]
    l = W.shape[1]

    Xt = X.transpose()
    XtW = np.matmul(Xt, W)

    # if b != 0:
    #     # pumping into matrix with shape (m,l)
    #     B = pump(b, l, m, False)
    # else:
    #     B = 0
    #
    # XtW = XtW - B
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


def softmax_objective_single(X, W, C, b=0):
    l = W.shape[1]
    XtW = np.matmul(X, W)
    eta = XtW.max()
    num = np.exp(XtW - eta)
    # summing rows to get denominator
    den = num.sum()
    logDXtW = np.log(num / den)
    res = 0
    for i in range(l):
        res -= sum(C[i, :]*logDXtW[:, i])
    return res


def softmax_gradient_single(X, W, C, b=0):

    XtW = np.matmul(X,W)
    expXtW = np.exp(XtW)
    D = 1/(expXtW.sum())
    DexpXtw = D * expXtW

    return np.outer(X, DexpXtw-C)



# computes the softmax gradient for all weight vectors together
def softmax_gradient(X, W, C, b=0):
    m = X.shape[1]
    Xt = X.transpose()

    XtW = np.matmul(Xt,W)
    expXtW = np.exp(XtW)
    D = np.diag(1/(expXtW.sum(axis=1)))
    DexpXtw = np.matmul(D, expXtW)

    return (1/m)*np.matmul(X, DexpXtw-C.transpose())


# an envelope function used in gradient testing
def func(X, W, C, w):
    W1 = np.zeros(W.shape)
    W1 += W
    W1[:, 0] = w
    return softmax_objective(X, W1, C)


# an envelope function used in gradient testing
def gradient(X, W, C, w):
    W1 = np.zeros(W.shape)
    W1 += W
    W1[:, 0] = w
    return softmax_gradient(X, W, C)


def stochastic_gradient_descent(X, W, C, max_iter=100, learning_rate=0.02, batch_size = 10):

    history = [W]

    for i in range(max_iter):
        num_of_mini_batches = round(X.shape[1] / batch_size)
        perm = np.random.permutation(X.shape[1])

        for j in range(num_of_mini_batches):
            batch_indexes = perm[(j * batch_size):((j + 1) * batch_size)];
            # iterating over all mini batches
            mini_batch_X = X[:,batch_indexes]
            mini_batch_C = C[:,batch_indexes]
            # iterate over the mini_batch [previous_index, ... next_index]

            # grad = 0
            # for l in range(batch_size):
            #     grad += softmax_gradient_single(mini_batch_X[:,l], W, mini_batch_C[:,l])
            # grad = (1/batch_size)*grad

            grad = softmax_gradient(mini_batch_X, W, mini_batch_C)

            learning_rate = 1/(j+1)

            W = W - learning_rate * grad

        print(softmax_objective(X, W, C))
        history.append(W)

    return history, W


def predict(W, X):
    prob = np.matmul(X.transpose(),W)
    res = prob.argmax(axis=1)
    return  res

if __name__ == "__main__":


    # GMMData = scipy.io.loadmat('GMMData.mat')
    PeaksData = scipy.io.loadmat('PeaksData.mat')
    # SwissRollData = scipy.io.loadmat('SwissRollData.mat')

    # Ct = GMMData['Ct']
    # Cv = GMMData['Cv']
    # Yt = GMMData['Yt']
    # Yv = GMMData['Yv']

    Ct = PeaksData['Ct']
    Cv = PeaksData['Cv']
    Yt = PeaksData['Yt']
    Yv = PeaksData['Yv']

    # Ct = SwissRollData['Ct']
    # Cv = SwissRollData['Cv']
    # Yt = SwissRollData['Yt']
    # Yv = SwissRollData['Yv']



    print(Ct.shape)

    print(Cv.shape)

    print(Yt.shape)

    print(Yv.shape)

    n = Yv.shape[0]
    l = Cv.shape[0]


    W = np.ones((n,l))*10

    history, W = stochastic_gradient_descent(Yv, W, Cv)





    # C = predict(W, SwissYv)
    # print((C-SwissCv).sum(axis=0))


    print(W)





    # A = np.asarray([[1,1,4],[1,1,1]])
    # W = np.asarray([[0,0,1,0,0,0],[0,0,0,0,0,1]])
    # C = np.asarray([[0,0,0],[0,0,0],[1,0,0],[0,0,1],[0,0,0],[0,1,0]])
    #
    # print(softmax_objective(A, W, C))
    # Gradient testing
    # X = np.random.rand(3, 5)
    # W = np.random.rand(3, 3)
    # c = np.random.randint(0, 3, 5)
    # C = np.asarray([c == 0, c == 1, c == 2])
    # w = W[:, 0]
    #
    # d = np.random.rand(3)
    # epsilon = 1
    #
    # f = lambda w: func(X, W, C, w)
    # gradf = lambda w: gradient(X, W, C, w)
    #
    # curr = f(w)
    # last = f(w)
    #
    # for i in range(10):
    #     last = curr
    #     epsilon = epsilon*0.5
    #     curr = abs(f(w+epsilon*d)-f(w))
    #     print('ratio 1 is: ', curr/last)
    #
    # d = np.random.rand(3)
    # epsilon = 0.2
    #
    # curr = f(w)
    # last = f(w)
    #
    # for i in range(10):
    #     last = curr
    #     epsilon = epsilon * 0.5
    #     curr = abs(f(w + epsilon * d) - f(w)-epsilon*sum(d*(gradf(w)[:, 0])))
    #     print('ratio 2 is: ', curr / last)








