import numpy as np
import stochastic_gradient_descent as sgd
import gradients as grad


# an envelope function used in gradient testing
def func(X, W, C, v, param, column=0):
    if param == 'w':
        W1 = np.zeros(W.shape)
        W1 += W
        W1[:, column] = v
        X1 = X
    elif param == 'x':
        X1 = np.zeros(X.shape)
        X1 += X
        X1[:, column] = v
        W1 = W
    else:
        return None

    return sgd.softmax_objective(X1, W1, C)


# an envelope function used in gradient testing
def grad(X, W, C, v, param, column=0):
    if param == 'w':
        W1 = np.zeros(W.shape)
        W1 += W
        W1[:, column] = v
        X1 = X
    elif param == 'x':
        X1 = np.zeros(X.shape)
        X1 += X
        X1[:, column] = v
        W1 = W
    else:
        return None

    return sgd.softmax_objective(X1, W1, C)


def generate_random_matrices(data_dim, num_of_samples, num_of_classes):
    X = np.random.rand(data_dim, num_of_samples)
    W = np.random.rand(data_dim, num_of_samples)
    c = np.random.randint(0, num_of_classes, num_of_samples)

    C = []
    for i in range(num_of_classes):
        C.append(c == i)
    C = np.asarray(C)

    return X, W, C


def test_ratios(f, gradf, x, iterations=10, epsilon=1, column=0):
    d = np.random.rand(x.shape[0])

    curr = f(x)

    for i in range(iterations):
        last = curr
        epsilon = epsilon * 0.5
        curr = abs(f(x + epsilon * d) - f(x))
        print('ratio 1 is: ', curr / last)


    curr = f(x)

    for i in range(iterations):
        last = curr
        epsilon = epsilon * 0.5
        curr = abs(f(x + epsilon * d) - f(x) - epsilon * sum(d * gradf(x)[:, column]))
        print('ratio 2 is: ', curr / last)


def gradient_test_x(data_dim, num_of_samples, num_of_classes, iterations=10, epsilon=1, column=0):
    print("gradient testing by x")

    X, W, C = generate_random_matrices(data_dim, num_of_samples, num_of_classes)

    x = X[:, column]

    f = lambda x: func(X, W, C, x, 'x')
    gradf = lambda x: grad(X, W, C, x, 'x')

    test_ratios(f, gradf, x, iterations, epsilon, column)

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
    f = lambda w: func(X, W, C, w, 'w', column)
    gradf = lambda w: grad(X, W, C, w, 'w', column)
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
