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
    n = (np.shape(X)[0])
    Xtranspose = np.transpose(X)
    sigXw = vector_sigmoid(np.matmul(Xtranspose, w))
    c1 = (Y > 0)
    c2 = (c1 == False)
    veclog = np.vectorize(log)
    # print(np.matmul(Xtranspose, w))
    # print(sigXw)
    # print(veclog(sigXw))
    c1_log_sigXw = np.matmul(c1, veclog(sigXw))
    c2_log_sigXw = np.matmul(c2, veclog(np.ones(m)-sigXw))

    return -(1/m)*(c1_log_sigXw + c2_log_sigXw)


def logistic_gradient(X, Y, w):
    m = (np.shape(Y)[0])
    # n = (np.shape(X)[0])
    c1 = (Y > 0)
    # c2 = (c1 == False)
    # sig = np.vectorize(sigmoid)
    sigXw = vector_sigmoid(np.matmul(np.transpose(X), w))

    return (1/m)*np.matmul(X,sigXw-c1)

def logistic_hessian(X, Y, w):
    n = (np.shape(X)[0])
    m = (np.shape(Y)[0])
    Xtranspose = np.transpose(X)
    sigXw = vector_sigmoid(np.matmul(Xtranspose, w))
    D = np.diag(sigXw * (np.ones(m)-sigXw))

    return np.matmul(np.matmul(X, D),Xtranspose)/m


if __name__ == "__main__":
    # X = np.array([[-1,1,1,4,8],\
    #               [1,-2,3,6,2],\
    #               [1,9,5,-7,0],\
    #               [2,-5,8,3,2],\
    #               [-43,5,6,-3,5],\
    #               ])
    # Y = np.array([0,0,1,0,1,0,0,0])
    # w = np.array([0, 0, 1, 0, 1, 0, 0, 0])


    X = np.random.rand(5,8)
    w = np.random.rand(5)
    negate = np.random.rand(5)
    negate = negate < 0.5
    negate = (-1 * negate) + (negate == False)
    w = w * negate
    # Y = vector_sigmoid(np.matmul(np.transpose(X), w))

    Y = np.random.rand(8)

    Y = Y > 0.5


    # logistic_gradient(X, Y, w)
    # logistic_hessian(X, Y, w)
    # logistic_objective(X, Y, w)

    d = np.random.rand(5)
    epsilon = 0.2
    f = lambda w: logistic_objective(X,Y,w)
    gradf = lambda w: logistic_gradient(X,Y,w)
    # print(f(w+epsilon*d))
    # print(f(w))
    print(abs(f(w+epsilon*d)-f(w)))
    print((abs(f(w+epsilon*d)-f(w)))**2)
    print(abs(f(w + epsilon * d) - f(w) - epsilon * np.matmul(d, gradf(w))))
