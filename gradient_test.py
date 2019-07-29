import numpy as np
import stochastic_gradient_descent as sgd
import gradients as grads
from scipy.linalg import norm


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
    X = np.random.randn(data_dim, num_of_samples)
    W = np.random.randn(data_dim, num_of_classes)
    b = np.random.randn(num_of_classes, 1)
    c = np.random.randint(0, num_of_classes, num_of_samples)

    C = []
    for i in range(num_of_classes):
        C.append(c == i)
    C = np.asarray(C)

    return X, W, C, b


def test_ratios(f, gradf, x, iterations=10, init_epsilon=1, column=0):
    d = np.random.rand(x.shape[0])
    curr = f(x)
    epsilon = init_epsilon

    for i in range(iterations):
        last = curr
        epsilon = epsilon * 0.5
        x1 = x + epsilon * d
        curr = abs(f(x1) - f(x))
        x = x1
        print('ratio 1 is: ', curr / last)

    d = np.random.rand(x.shape[0])
    curr = f(x)
    epsilon = init_epsilon

    for i in range(iterations):
        last = curr
        epsilon = epsilon * 0.5
        x1 = x + epsilon * d
        curr = abs(f(x1) - f(x) - epsilon * sum(d * gradf(x)[:-1, column]))
        x = x1
        print('ratio 2 is: ', curr / last)


def gradient_test_x(data_dim, num_of_samples, num_of_classes, iterations=10, epsilon=1, column=0):
    print("gradient testing by x")

    X, W, C, b = generate_random_matrices(data_dim, num_of_samples, num_of_classes)

    W = np.random.randn(data_dim+1, num_of_classes)

    x = X[:, column]

    f = lambda x: func(X, W, C, x, 'x')
    gradf = lambda x: grad(X, W, C, x, 'x')

    test_ratios(f, gradf, x, iterations, epsilon, column)


def gradient_test_w(data_dim, num_of_samples, num_of_classes, iterations=10, epsilon=1, column=0):
    print("gradient testing by w")

    X, W, C, b = generate_random_matrices(data_dim, num_of_samples, num_of_classes)

    W = np.random.randn(data_dim+1, num_of_classes)

    w = W[:, column]

    f = lambda w: func(X, W, C, w, 'w')
    gradf = lambda w: grad(X, W, C, w, 'w')

    test_ratios(f, gradf, w, iterations, epsilon, column)


def simulate_propogation(X, W, b, replace=False, param = None, y=None, column=0):
    X1 = X
    W1 = W
    b1 = b

    # replacing parameters if necessary
    if replace:
        if param == 'x':
            X1[:, column] = y
        elif param == 'w':
            W1[:, column] = y
        elif param == 'b':
            b1 = y
        else:
            return None

    # print(b1.reshape(b.shape[0], 1), 'b')
    # print(np.matmul(W1.transpose(), X1), 'Wx')

    layer_res = np.matmul(W1.transpose(), X1) + b1

    return np.tanh(layer_res)
    # return ReLU2(layer_res)


def test_jac_b_t_v(data_dim, num_of_samples, num_of_classes, iterations=10, epsilon=1):

    f = lambda z: np.tanh(np.matmul(W.transpose(), X) + z)

    # Generating random inputs for layer
    X, W, C, b = generate_random_matrices(data_dim, num_of_samples, num_of_classes)
    d = np.random.randn(*b.shape)
    eps = epsilon
    curr = norm(f(b))

    for i in range(iterations):
        last = curr
        eps *= 0.5
        b_new = b + eps * d
        curr = norm(f(b_new) - f(b))
        b = b_new
        print('ratio 1 is: ', curr / last)

    X, W, C, b = generate_random_matrices(data_dim, num_of_samples, num_of_classes)
    d = np.random.randn(*b.shape)
    eps = epsilon
    curr = norm(f(b))
    der = 1 - f(b)**2

    for i in range(iterations):
        last = curr
        eps *= 0.5
        b_new = b + eps * d
        curr = norm(f(b_new) - f(b) - grads.JacV_b(der, eps * d))
        b = b_new
        der = 1 - f(b) ** 2
        print('ratio 2 is: ', curr / last)


def test_jac_w_t_v(data_dim_1, data_dim_2, num_of_samples, iterations=10, epsilon=1):

    # Generating random inputs for layer
    X, W, C, b = generate_random_matrices(data_dim_1, num_of_samples, data_dim_2)

    f = lambda z: np.tanh(np.matmul(z.transpose(), X) + b)

    d = np.random.randn(data_dim_1, data_dim_2)
    eps = epsilon
    curr = norm(f(W))


    for i in range(iterations):
        last = curr
        eps *= 0.5
        W_new = W + eps * d
        curr = norm(f(W_new) - f(W))
        print('ratio 1 is: ', curr / last)
        W = W_new

    X, W, C, b = generate_random_matrices(data_dim_1, num_of_samples, data_dim_2)

    f = lambda z: np.tanh(np.matmul(z.transpose(), X) + b)

    # d = np.random.randn(data_dim, data_dim)
    d = np.random.randn(data_dim_1, data_dim_2)
    # d[:, 0] = d[:,0] + np.random.randn(data_dim)
    eps = epsilon
    curr = norm(f(W))
    der = 1 - f(W)**2

    for i in range(iterations):
        last = curr
        eps *= 0.5
        W_new = W + eps * d
        # curr = norm(f(W_new) - f(W) - grads.JacV_w(X, der, eps * d))

        # print('W dim', W.shape)
        # print('d dim', d.shape)
        # print('X dim', X.shape)
        # print('der dim', der.shape)
        # print('derX dim', np.matmul(der, X.transpose()).shape)

        # this works when W and X are single vectors.
        # curr = norm(f(W_new) - f(W) - eps * der * np.matmul(d.transpose(), X))
        # curr = norm(f(W_new) - f(W) - np.matmul(der * X.transpose(), eps * d))
        curr = norm(f(W_new) - f(W) - grads.JacV_w(X, der, eps * d))

        W = W_new
        der = 1 - f(W) ** 2
        print('ratio 2 is: ', curr / last)


def test_jac_x_t_v(data_dim_1, data_dim_2, num_of_samples, iterations=10, epsilon=1):

    # Generating random inputs for layer
    X, W, C, b = generate_random_matrices(data_dim_1, num_of_samples, data_dim_2)

    f = lambda z: np.tanh(np.matmul(W.transpose(), z) + b)

    d = np.random.randn(data_dim_1, num_of_samples)
    eps = epsilon
    curr = norm(f(X))

    for i in range(iterations):
        last = curr
        eps *= 0.5
        X_new = X + eps * d
        curr = norm(f(X_new) - f(X))
        print('ratio 1 is: ', curr / last)
        X = X_new

    X, W, C, b = generate_random_matrices(data_dim_1, num_of_samples, data_dim_2)

    f = lambda z: np.tanh(np.matmul(W.transpose(), z) + b)

    print(f(X).shape)

    d = np.random.randn(data_dim_1, num_of_samples)
    eps = epsilon
    curr = norm(f(X))
    der = 1 - f(X)**2

    for i in range(iterations):

        last = curr
        eps *= 0.5
        X_new = X + eps * d

        curr = norm(f(X_new) - f(X) - grads.JacV_x(W, der, eps * d))

        X = X_new
        der = 1 - f(X) ** 2
        print('ratio 2 is: ', curr / last)


def ReLU2(X):
    X[X < 0] = 0
    return X
