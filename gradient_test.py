import numpy as np
import stochastic_gradient_descent as sgd
import gradients as grads


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
        return grads.softmax_gradient(X1, W1, C)
    elif param == 'x':
        X1 = np.zeros(X.shape)
        X1 += X
        X1[:, column] = v
        W1 = W
        return grads.softmax_data_gradient(X1, W1, C)
    else:
        return None


def generate_random_matrices(data_dim, num_of_samples, num_of_classes):
    X = np.random.rand(data_dim, num_of_samples)
    W = np.random.rand(data_dim, num_of_classes)
    c = np.random.randint(0, num_of_classes, num_of_samples)

    C = []
    for i in range(num_of_classes):
        C.append(c == i)
    C = np.asarray(C)

    return X, W, C


def test_ratios(f, gradf, x, iterations=10, init_epsilon=1, column=0):
    d = np.random.rand(x.shape[0])
    curr = f(x)
    epsilon = init_epsilon

    for i in range(iterations):
        last = curr
        epsilon = epsilon * 0.5
        curr = abs(f(x + epsilon * d) - f(x))
        print('ratio 1 is: ', curr / last)

    d = np.random.rand(x.shape[0])
    curr = f(x)
    epsilon = init_epsilon

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


def gradient_test_w(data_dim, num_of_samples, num_of_classes, iterations=10, epsilon=1, column=0):
    print("gradient testing by w")

    X, W, C = generate_random_matrices(data_dim, num_of_samples, num_of_classes)

    w = W[:, column]

    f = lambda w: func(X, W, C, w, 'w')
    gradf = lambda w: grad(X, W, C, w, 'w')

    test_ratios(f, gradf, w, iterations, epsilon, column)

