import scipy.io
import numpy as np
from matplotlib import pyplot
from scipy.linalg import norm
import mnist_handler


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
        res += sum(C[i, :]*logDXtW[:, i])

    return res*(-1/m)


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

    logDXtW = np.log(num / den)

    DexpXtw = num / den

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


def stochastic_gradient_descent(X, W, C, max_iter=10000, learning_rate=0.02, batch_size = 1000):

    history = []

    for i in range(max_iter):
        num_of_mini_batches = round(X.shape[1] / batch_size)
        perm = np.random.permutation(X.shape[1])

        learning_rate = 1 / np.sqrt(i + 1)

        for j in range(num_of_mini_batches):
            batch_indexes = perm[(j * batch_size):((j + 1) * batch_size)];
            # iterating over all mini batches
            mini_batch_X = X[:, batch_indexes]
            mini_batch_C = C[:, batch_indexes]

            # iterate over the mini_batch [previous_index, ... next_index]

            # grad = 0
            # for l in range(batch_size):
            #     grad += softmax_gradient_single(mini_batch_X[:,l], W, mini_batch_C[:,l])
            # grad = (1/batch_size)*grad

            grad = softmax_gradient(mini_batch_X, W, mini_batch_C)

            W = W - learning_rate * grad

        if i % 100 == 0:
            print(i)
            print(softmax_objective(X, W, C))

        # print(softmax_objective(X, W, C))
        history.append(softmax_objective(X, W, C))

    return history, W


def predict(W, X):
    prob = np.matmul(X.transpose(),W)
    res = prob.argmax(axis=1)
    return res

if __name__ == "__main__":


    # GMMData = scipy.io.loadmat('GMMData.mat')
    # Ct = GMMData['Ct']
    # Cv = GMMData['Cv']
    # Yt = GMMData['Yt']
    # Yv = GMMData['Yv']

    # PeaksData = scipy.io.loadmat('PeaksData.mat')
    # Ct = PeaksData['Ct']
    # Cv = PeaksData['Cv']
    # Yt = PeaksData['Yt']
    # Yv = PeaksData['Yv']

    # SwissRollData = scipy.io.loadmat('SwissRollData.mat')
    # Ct = SwissRollData['Ct']
    # Cv = SwissRollData['Cv']
    # Yt = SwissRollData['Yt']
    # Yv = SwissRollData['Yv']

    # n = Yt.shape[0]
    # l = Ct.shape[0]
    #
    # # adding bias
    # m = Yt.shape[1]
    # bias_row = np.ones(m)
    # Yt = np.vstack([Yt, bias_row])
    #
    # m = Yv.shape[1]
    # bias_row = np.ones(m)
    # Yv = np.vstack([Yv, bias_row])
    #
    # print(Ct.shape)
    #
    # print(Cv.shape)
    #
    # print(Yt.shape)
    #
    # print(Yv.shape)
    #
    # X = Yv
    # C = Cv
    #
    # n = X.shape[0]
    # l = C.shape[0]
    #
    # W = np.ones((n, l))
    #
    # history, W = stochastic_gradient_descent(X, W, C)
    #
    # print(history[-1])
    #
    #
    #
    # C = predict(W, Yv)
    # # print((C-Cv).sum(axis=0))
    #
    # C = np.vstack([C,Cv])
    #
    # print(W)





    # reading the training data
    Y = mnist_handler.read_label_file(
        'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\train-labels.idx1-ubyte')
    X = mnist_handler.read_image_file(
        'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\train-images.idx3-ubyte')


    print(X.shape)
    print(Y.shape)


    C = np.zeros((10, Y.shape[0]))

    # creating C as required
    for i in range(10):
        C[i, :] = Y == i

    print(C.shape)

    X.shape = (X.shape[0], X.shape[1] * X.shape[2])

    print(X.shape)

    X1 = X.transpose()

    X = X1*(1/255)

    m = X.shape[1]
    n = X.shape[0]
    l = C.shape[0]

    bias_row = np.ones(m)

    X = np.vstack([X,bias_row])

    W = np.zeros((n+1, l))

    history, W = stochastic_gradient_descent(X, W, C)







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
    #







