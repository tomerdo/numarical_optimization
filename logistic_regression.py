from math import exp
from math import log
import numpy as np

def sigmoid(x):
    return 1/(1+exp(-x))

def vector_sigmoid(X):
    n = (np.shape(X))[0]
    res = np.zeros(n)
    for i in range(n):
        res[i] = sigmoid(X[i])
    return res


def logistic_objective(X, Y, w):
    m = (np.shape(Y)[0])
    n = (np.shape(X)[1])
    Xtranspose = np.transpose(X)
    sigXw = vector_sigmoid(np.matmul(Xtranspose, w))
    c1 = (Y > 0)
    c2 = (c1 == False)
    veclog = np.vectorize(log)
    c1_log_sigXw = np.matmul(c1, veclog(sigXw))
    c2_log_sigXw = np.matmul(c1, np.ones(n)-veclog(sigXw))

    return -(1/m)*(c1_log_sigXw + c2_log_sigXw)


def logistic_gradient(X, Y, w):
    m = (np.shape(Y)[0])
    # n = (np.shape(X)[1])
    c1 = (Y > 0)
    # c2 = (c1 == False)
    # sig = np.vectorize(sigmoid)
    sigXw = vector_sigmoid(np.matmul(np.transpose(X), w))

    return np.matmul(X,sigXw-c1)/m

def logistic_hessian(X, Y, w):
    n = (np.shape(X)[1])
    m = (np.shape(Y)[0])
    Xtranspose = np.transpose(X)
    sigXw = vector_sigmoid(np.matmul(Xtranspose, w))
    D = np.diag(sigXw * (np.ones(n)-sigXw))

    return np.matmul(np.matmul(X, D),Xtranspose)/m


if __name__ == "__main__":
    X = np.array([[1,1,1,4,8],\
                  [1,2,3,6,2],\
                  [1,9,5,7,0],\
                  [2,5,8,3,2],\
                  [43,5,6,3,5],\
                  ])
    Y = np.array([0,0,1,0,1])
    w = np.array([2,3,1,1,5])
    logistic_gradient(X, Y, w)
    logistic_hessian(X, Y, w)
    logistic_objective(X, Y, w)
    d = np.random.rand(5, 1)

