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
    b = np.random.randn(num_of_classes)
    c = np.random.randint(0, num_of_classes, num_of_samples)

    C = []
    for i in range(num_of_classes):
        C.append(c == i)
    C = np.asarray(C)

    return X, W, C, b


def test_ratios(f, gradf, x, iterations=17, init_epsilon=1, column=0):
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
        curr = abs(f(x1) - f(x) - epsilon * sum(d * gradf(x)[:, column]))
        x = x1
        print('ratio 2 is: ', curr / last)


def gradient_test_x(data_dim, num_of_samples, num_of_classes, iterations=10, epsilon=1, column=0):
    print("gradient testing by x")

    X, W, C, b = generate_random_matrices(data_dim, num_of_samples, num_of_classes)

    x = X[:, column]

    f = lambda x: func(X, W, C, x, 'x')
    gradf = lambda x: grad(X, W, C, x, 'x')

    test_ratios(f, gradf, x, iterations, epsilon, column)


def gradient_test_w(data_dim, num_of_samples, num_of_classes, iterations=10, epsilon=1, column=0):
    print("gradient testing by w")

    X, W, C, b = generate_random_matrices(data_dim, num_of_samples, num_of_classes)

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
            X1[:,column] = y
        elif param == 'W':
            X1[:,column] = y
        else:
            b1 = y

    layer_res = np.matmul(W1.transpose(), X1) + b1.reshape(b.shape[0],1)
    layer_res[layer_res < 0] = 0
    relu_der = layer_res > 0

    return relu_der, layer_res

def test_jac_t_v(jac_f, data_dim, num_of_samples, num_of_classes, param, iterations=10, epsilon=1, column=0):
    # Generating random inputs for layer
    X, W, C, b = generate_random_matrices(data_dim, num_of_samples, num_of_classes)

    # Simulating forward propogation
    relu_der, next_x = simulate_propogation(X, W, b)


    f = lambda x: simulate_propogation(X, W, b, True, param, x)

    d = np.random.randn(num_of_classes)
    eps = epsilon

    curr = norm(f(b)[1].sum(axis=1)/num_of_samples)

    for i in range(iterations):
        last = curr
        eps *= 0.5
        b1 = b + eps * d
        curr = norm(f(b1)[1].sum(axis=1)/num_of_samples - f(b)[1].sum(axis=1)/num_of_samples)
        b = b1
        print('ratio 1 is: ', curr / last)



    X, W, C, b = generate_random_matrices(data_dim, 1, num_of_classes)

    b.shape = num_of_classes

    # g = lambda b: np.tanh(np.matmul(W.transpose(), X).reshape(num_of_classes) + b)
    #
    # relu_der = 1-(np.tanh(np.matmul(W.transpose(), X).reshape(num_of_classes) + b))**2

    g = lambda k: ReLU2(np.matmul(W.transpose(), X).reshape(num_of_classes) + k)

    c = g(b)
    relu_der = c > 0

    d = np.random.randn(num_of_classes)
    eps = epsilon


    # print(np.matmul(W.transpose(), X))
    # print(g(b))
    # print(relu_der)

    curr = norm(g(b))

    for i in range(iterations):
        last = curr
        eps *= 0.5
        b1 = b + eps * d
        # print(relu_der)
        # print(d)
        # print(relu_der*eps*d)
        # print(g(b1))
        # print(np.matmul(np.diag(relu_der), eps*d))
        curr = norm(g(b1) - g(b) - np.matmul(np.diag(relu_der), eps*d))
        b = b1
        print('ratio 2 is: ', curr / last)
        c = g(b)
        relu_der = c > 0



def ReLU2(X):
    X[X < 0] = 0
    return X