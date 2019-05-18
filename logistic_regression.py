from math import exp
from math import log
from scipy.linalg import norm
import numpy as np
from numpy.linalg import inv
import mnist_handler
from matplotlib import pyplot


def sigmoid(x):
    # print('sig')
    # print(x)
    return 1/(1+exp(-x))


# the sigmoid function that operates on a vector (sigmoid on every entry)
def vector_sigmoid(X):
    n = (np.shape(X))[0]
    res = np.zeros(n)
    for i in range(n):
        res[i] = sigmoid(X[i])
    return res


# the log function that operates on a vector (log on every entry)
def vector_log(X):
    n = (np.shape(X))[0]
    res = np.zeros(n)
    for i in range(n):
        # print('log')
        # print(X[i])
        res[i] = log(X[i])
    return res


# the objective function of logistic regression
def logistic_objective(X, Y, w):
    m = (np.shape(Y)[0])
    n = (np.shape(X)[0])
    Xtranspose = np.transpose(X)

    # for debugging purposes
    zzz = np.matmul(Xtranspose, w)

    sigXw = vector_sigmoid(np.matmul(Xtranspose, w))
    c1 = (Y > 0)
    c2 = (c1 == False)
    # veclog = np.vectorize(log)

    c1_log_sigXw = np.matmul(c1, vector_log(sigXw))
    c2_log_sigXw = np.matmul(c2, vector_log(1-sigXw))

    return -(1/m)*(c1_log_sigXw + c2_log_sigXw)


# the gradient of logistic regression
def logistic_gradient(X, Y, w):
    m = (np.shape(Y)[0])
    # n = (np.shape(X)[0])
    c1 = (Y > 0)
    # c2 = (c1 == False)
    # sig = np.vectorize(sigmoid)

    sigXw = vector_sigmoid(np.matmul(np.transpose(X), w))

    return (1/m)*np.matmul(X, sigXw-c1)


# the Hessian of logistic regression
def logistic_hessian(X, Y, w):
    n = (np.shape(X)[0])
    m = (np.shape(Y)[0])
    Xtranspose = np.transpose(X)
    sigXw = vector_sigmoid(np.matmul(Xtranspose, w))
    D = np.diag(sigXw * (np.ones(m)-sigXw))

    return (1/m)*np.matmul(np.matmul(X, D),Xtranspose)


# finds a stepsize alpha using the Armijo method for steepest descent
def GD_line_search(A, b, x, grad, fx, alpha0 = 0.00001, beta = 0.5, c = 0.0001, iterations = 10):

    alphaj = alpha0
    flag = True
    iter = 0


    while flag:
        res = logistic_objective(A, b, x - alphaj*grad) - fx - c * alphaj * sum(grad * grad)
        iter += 1
        if res <= 0 or iter >= iterations:
            flag = False
        alphaj *= beta

    return alphaj


# finds a stepsize alpha using the Armijo method for the exact Newton method
def newton_line_search(A, b, x, grad, dn, fx, alpha0 = 0.00007, beta = 0.5, c = 0.0001, iterations = 10):

    alphaj = alpha0
    flag = True
    iter = 0


    while flag:
        res = logistic_objective(A, b, x - alphaj*dn) - fx - c * alphaj * sum(dn * grad)
        iter += 1
        if res <= 0 or iter >= iterations:
            flag = False
        alphaj *= beta

    return alphaj


def mnist_GD(A, b, x=None, iterations=100, tolerance=0.1, criteria=0):

    if x is None:
        n = (A.shape)[0]
        x = np.zeros(n)

    plot1 = []
    plot2 = []

    for iter in range(iterations):
        fx = logistic_objective(A, b, x)
        grad = logistic_gradient(A, b, x)
        alpha = GD_line_search(A, b, x, grad, fx)
        x = x - alpha * grad
        plot1.append(fx)

    return x, plot1, plot2


def mnist_newton(A, b, x=None, iterations=1000, tolerance=0.1, criteria=0):

    if x is None:
        n = (A.shape)[0]
        x = np.zeros(n)

    # these lists will eventually hold the data that needs to be plotted
    plot1 = []
    plot2 = []

    epsilon = 10**(-5)

    for iter in range(iterations):
        fx = logistic_objective(A, b, x)
        grad = logistic_gradient(A, b, x)
        hess = logistic_hessian(A, b, x)
        # inv_hess = inv(logistic_hessian(A, b, x))
        inv_hess = inv(np.diag(np.diag(hess)) + epsilon * np.eye(n))
        dn = np.matmul(inv_hess, grad)
        alpha = newton_line_search(A, b, x, grad, dn, fx)
        x = x - alpha * dn
        plot1.append(fx)

    return x, plot1, plot2


if __name__ == "__main__":
    # steepest descent for digits 0 1
    if True:
        1
        # if True:
        #     # ==================================================================
        #     # ================   This is the learning phase ====================
        #     # ==================================================================
        #
        #     # reading the training data
        #     Y = mnist_handler.read_label_file(
        #         'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\train-labels.idx1-ubyte')
        #     X = mnist_handler.read_image_file(
        #         'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\train-images.idx3-ubyte')
        #
        #     # getting the digits we need
        #     X01, Y01 = mnist_handler.get_digits([0, 1], X, Y)
        #
        #     # transforming the matrices representing the images into vectors of pixels
        #     X01.shape = (np.shape(X01)[0], np.shape(X01)[1] * np.shape(X01)[2])
        #
        #     # make matrix of order n x m instead of m x n
        #     X01 = np.transpose(X01)
        #
        #     # making the vector boolean
        #     Y01 = Y01 > 0
        #
        #     x, plot1, plot2 = mnist_GD(X01, Y01)
        #
        #     pyplot.plot(plot1)
        #     pyplot.show()
        #
        #     pred = np.transpose(X01)
        #
        #     # ==================================================================
        #     # ================   This is the testing phase =====================
        #     # ==================================================================
        #
        #     # reading the test data
        #     Ytest = mnist_handler.read_label_file(
        #         'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-labels.idx1-ubyte')
        #     Xtest = mnist_handler.read_image_file(
        #         'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-images.idx3-ubyte')
        #
        #     # getting the digits we need
        #     Xtest01, Ytest01 = mnist_handler.get_digits([0, 1], Xtest, Ytest)
        #
        #     # transforming the matrices representing the images into vectors of pixels
        #     Xtest01.shape = (np.shape(Xtest01)[0], np.shape(Xtest01)[1] * np.shape(Xtest01)[2])
        #
        #     # using the output x to predict the results
        #     ans = np.matmul(Xtest01, x)
        #
        #     # if the result is bigger than 0.5 we predict 1. otherwise we predict 0.
        #     ans = ans>0.5
        #
        #     # looking for mistakes
        #     diff = ans != Ytest01
        #
        #     print(sum(diff))

    # steepest descent for digits 8 9
    if True:
        1
        # if True:
        #     # ==================================================================
        #     # ================   This is the learning phase ====================
        #     # ==================================================================
        #
        #     # reading the training data
        #     Y = mnist_handler.read_label_file(
        #         'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\train-labels.idx1-ubyte')
        #     X = mnist_handler.read_image_file(
        #         'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\train-images.idx3-ubyte')
        #
        #     # getting the digits we need
        #     X89, Y89 = mnist_handler.get_digits([8, 9], X, Y)
        #
        #     # transforming the matrices representing the images into vectors of pixels
        #     X89.shape = (np.shape(X89)[0], np.shape(X89)[1] * np.shape(X89)[2])
        #
        #     # make matrix of order n x m instead of m x n
        #     X89 = np.transpose(X89)
        #
        #     # making the vector boolean
        #     Y89 = Y89 > 8
        #
        #     x, plot1, plot2 = mnist_GD(X89, Y89)
        #
        #     pyplot.plot(plot1)
        #     pyplot.show()
        #
        #     pred = np.transpose(X89)
        #
        #     # ==================================================================
        #     # ================   This is the testing phase =====================
        #     # ==================================================================
        #
        #     # reading the test data
        #     Ytest = mnist_handler.read_label_file(
        #         'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-labels.idx1-ubyte')
        #     Xtest = mnist_handler.read_image_file(
        #         'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-images.idx3-ubyte')
        #
        #     # getting the digits we need
        #     Xtest89, Ytest89 = mnist_handler.get_digits([8, 9], Xtest, Ytest)
        #
        #     # transforming the matrices representing the images into vectors of pixels
        #     Xtest89.shape = (np.shape(Xtest89)[0], np.shape(Xtest89)[1] * np.shape(Xtest89)[2])
        #
        #     # using the output x to predict the results
        #     ans = np.matmul(Xtest89, x)
        #
        #     # if the result is bigger than 0.5 we predict 9. otherwise we predict 8.
        #     ans = ans>0.5
        #
        #     Ytest89 = Ytest89>8
        #
        #     # looking for mistakes
        #     diff = ans != Ytest89
        #
        #     print(sum(diff))

    # Newton descent for digits 8 9
    if True:
        1
        if True:
            # ==================================================================
            # ================   This is the learning phase ====================
            # ==================================================================

            # reading the training data
            Y = mnist_handler.read_label_file(
                'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\train-labels.idx1-ubyte')
            X = mnist_handler.read_image_file(
                'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\train-images.idx3-ubyte')

            # getting the digits we need
            X89, Y89 = mnist_handler.get_digits([8, 9], X, Y)

            # transforming the matrices representing the images into vectors of pixels
            X89.shape = (np.shape(X89)[0], np.shape(X89)[1] * np.shape(X89)[2])

            # make matrix of order n x m instead of m x n
            X89 = np.transpose(X89)

            # making the vector boolean
            Y89 = Y89 > 8

            x, plot1, plot2 = mnist_newton(X89, Y89)

            pyplot.plot(plot1)
            pyplot.show()

            pred = np.transpose(X89)

            # ==================================================================
            # ================   This is the testing phase =====================
            # ==================================================================

            # reading the test data
            Ytest = mnist_handler.read_label_file(
                'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-labels.idx1-ubyte')
            Xtest = mnist_handler.read_image_file(
                'D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-images.idx3-ubyte')

            # getting the digits we need
            Xtest89, Ytest89 = mnist_handler.get_digits([8, 9], Xtest, Ytest)

            # transforming the matrices representing the images into vectors of pixels
            Xtest89.shape = (np.shape(Xtest89)[0], np.shape(Xtest89)[1] * np.shape(Xtest89)[2])

            # using the output x to predict the results
            ans = np.matmul(Xtest89, x)

            # if the result is bigger than 0.5 we predict 9. otherwise we predict 8.
            ans = ans>0.5

            Ytest89 = Ytest89>8

            # looking for mistakes
            diff = ans != Ytest89

            print(sum(diff))


    # gradient and hessian tests
    if True:
        1
        # if True:
        #     X = np.random.rand(5,8)
        #     w = np.random.rand(5)
        #     negate = np.random.rand(5)
        #     negate = negate < 0.5
        #     negate = (-1 * negate) + (negate == False)
        #     w = w * negate
        #     # Y = vector_sigmoid(np.matmul(np.transpose(X), w))
        #
        #     Y = np.random.rand(8)
        #
        #     Y = Y > 0.5
        #     #
        #     #
        #     # logistic_gradient(X, Y, w)
        #     # logistic_hessian(X, Y, w)
        #     # logistic_objective(X, Y, w)
        #     #
        #     d = np.random.rand(5)
        #     epsilon = 0.2
        #     f = lambda w: logistic_objective(X,Y,w)
        #     gradf = lambda w: logistic_gradient(X,Y,w)
        #     hessf = lambda w: logistic_hessian(X, Y, w)
        #     # print(f(w+epsilon*d))
        #     # print(f(w))
        #
        #     last1 = 1
        #     last2 = 1
        #
        #     for i in range(10):
        #         epsilon = (0.5)**i
        #
        #         curr1 = abs(f(w+epsilon*d)-f(w))
        #         ratio1 = curr1/last1
        #
        #         curr2 = abs(f(w + epsilon * d) - f(w) - epsilon * np.matmul(d, gradf(w)))
        #         ratio2 = curr2 / last2
        #
        #         last1 = curr1
        #         last2 = curr2
        #
        #         print(ratio1)
        #         # print((abs(f(w+epsilon*d)-f(w)))**2)
        #         print(ratio2)
        #         print('\n')
        #
        #
        #     last1 = 1
        #     last2 = 1
        #
        #     for i in range(10):
        #         epsilon = (0.5) ** i
        #
        #         curr1 = norm(gradf(w + epsilon * d) - gradf(w))
        #         ratio1 = curr1/last1
        #
        #         curr2 = norm(gradf(w + epsilon * d) - gradf(w) - np.matmul(hessf(w), epsilon * d))
        #         ratio2 = curr2 / last2
        #
        #         last1 = curr1
        #         last2 = curr2
        #
        #         print(ratio1)
        #         # print((abs(f(w+epsilon*d)-f(w)))**2)
        #         print(ratio2)
        #
        #         # print((norm(gradf(w + epsilon * d) - gradf(w))) ** 2)
        #
        #         print('\n')


