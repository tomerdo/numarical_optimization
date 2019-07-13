import numpy as np
import stochastic_gradient_descent as sgd
import gradients as grad

# an envelope function used in gradient testing
def func_w(X, W, C, w):
    W1 = np.zeros(W.shape)
    W1 += W
    W1[:, 0] = w
    return sgd.softmax_objective(X, W1, C)


# an envelope function used in gradient testing
def gradient_w(X, W, C, w):
    W1 = np.zeros(W.shape)
    W1 += W
    W1[:, 0] = w
    return sgd.softmax_gradient(X, W1, C)


# an envelope function used in gradient testing
def func_x(X, W, C, x):
    X1 = np.zeros(X.shape)
    X1 += X
    X1[:, 0] = x
    return sgd.softmax_objective(X1, W, C)


# an envelope function used in gradient testing
def gradient_x(X, W, C, x):
    X1 = np.zeros(X.shape)
    X1 += X
    X1[:, 0] = x
    return grad.softmax_data_gradient(X1, W, C)


def gradient_test_x():
    global X, W, C
    # Gradient testing (x)
    print("gradient testing by x")
    X = np.random.rand(3, 5)
    W = np.random.rand(3, 3)
    c = np.random.randint(0, 3, 5)
    C = np.asarray([c == 0, c == 1, c == 2])
    x = X[:, 0]
    d = np.random.rand(3)
    epsilon = 1
    f = lambda x: func_x(X, W, C, x)
    gradf = lambda x: gradient_x(X, W, C, x)
    curr = f(x)
    last = f(x)
    for i in range(10):
        last = curr
        epsilon = epsilon * 0.5
        curr = abs(f(x + epsilon * d) - f(x))
        print('ratio 1 is: ', curr / last)
    d = np.random.rand(3)
    epsilon = 0.2
    curr = f(x)
    last = f(x)
    for i in range(10):
        last = curr
        epsilon = epsilon * 0.5
        curr = abs(f(x + epsilon * d) - f(x) - epsilon * sum(d * gradf(x)[:, 0]))
        print('ratio 2 is: ', curr / last)


def gradient_test_by_w():
    global W, C, X
    A = np.asarray([[1, 1, 4], [1, 1, 1]])
    W = np.asarray([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
    C = np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]])
    print(sgd.softmax_objective(A, W, C))
    # Gradient testing (w)
    print("gradient test by w")
    X = np.random.rand(3, 5)
    W = np.random.rand(3, 3)
    c = np.random.randint(0, 3, 5)
    C = np.asarray([c == 0, c == 1, c == 2])
    w = W[:, 0]
    d = np.random.rand(3)
    epsilon = 1
    f = lambda w: func_w(X, W, C, w)
    gradf = lambda w: gradient_w(X, W, C, w)
    curr = f(w)
    last = f(w)
    for i in range(10):
        last = curr
        epsilon = epsilon * 0.5
        curr = abs(f(w + epsilon * d) - f(w))
        print('ratio 1 is: ', curr / last)
    d = np.random.rand(3)
    epsilon = 0.2
    curr = f(w)
    last = f(w)
    for i in range(10):
        last = curr
        epsilon = epsilon * 0.5
        curr = abs(f(w + epsilon * d) - f(w) - epsilon * sum(d * (gradf(w)[:, 0])))
        print('ratio 2 is: ', curr / last)
