import scipy.io
import numpy as np
import mnist_handler
import matplotlib.pyplot as plt


# returns a matrix with n columns of copies of column vector vec of size m
def pump(vec, m, n, transpose=True):
    res = np.broadcast_to(vec, (n, m))
    if transpose:
        res = np.transpose(res)
    return res


def softmax_objective(X, W, C):
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
        res += sum(C[i, :] * logDXtW[:, i])

    return res * (-1 / m)


# computes the softmax gradient for all weight vectors (W) together
def softmax_gradient(X, W, C):
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

    DexpXtw = num / den

    return (1 / m) * np.matmul(X, DexpXtw - C.transpose())


# gradient by X used for back propagation
def softmax_data_gradient(X, W, C):
    m = X.shape[1]
    l = W.shape[1]

    Wt = W.transpose()

    WtX = np.matmul(Wt, X)

    eta = WtX.max(axis=1)
    # pumping into matrix with shape (m,l)
    eta = pump(eta, l, m)

    # num = np.exp(WtX - eta)
    num = np.exp(WtX)
    # summing rows to get denominator
    den = num.sum(axis=0)

    # pumping into matrix with shape (m,l)
    den = pump(den, m, l, False)

    DexpWtX = num / den

    return (1 / m) * np.matmul(W, DexpWtX - C)


# an envelope function used in gradient testing
def func_w(X, W, C, w):
    W1 = np.zeros(W.shape)
    W1 += W
    W1[:, 0] = w
    return softmax_objective(X, W1, C)


# an envelope function used in gradient testing
def gradient_w(X, W, C, w):
    W1 = np.zeros(W.shape)
    W1 += W
    W1[:, 0] = w
    return softmax_gradient(X, W1, C)


# an envelope function used in gradient testing
def func_x(X, W, C, x):
    X1 = np.zeros(X.shape)
    X1 += X
    X1[:, 0] = x
    return softmax_objective(X1, W, C)


# an envelope function used in gradient testing
def gradient_x(X, W, C, x):
    X1 = np.zeros(X.shape)
    X1 += X
    X1[:, 0] = x
    return softmax_data_gradient(X1, W, C)


# SGD - this algorithm takes the data,  the weights and the labels and it's learning the weights the optimize
# the softmax objective function
def stochastic_gradient_descent(X, W, C, x_valid=None, c_valid=None, max_iter=600, learning_rate=0.02,
                                batch_size=10_000, train_rate_data=[], validation_rate_data=[], epoch_data=[]):
    history = []

    # c_training = C
    # c_validation = c_valid
    # if not is_mnist_data:
        # converting labels to numeral form in order to calculate success rates
    labels = np.arange(C.shape[0])

    training_labels = pump(labels, C.shape[0], C.shape[1])
    c_training = (C * training_labels).sum(axis=0)

    validation_labels = pump(labels, c_valid.shape[0], c_valid.shape[1])
    c_validation = (c_valid * validation_labels).sum(axis=0)

    for i in range(max_iter):
        num_of_mini_batches = round(X.shape[1] / batch_size)
        perm = np.random.permutation(X.shape[1])

        learning_rate = 1 / np.sqrt(i + 1)

        for j in range(num_of_mini_batches):
            batch_indexes = perm[(j * batch_size):((j + 1) * batch_size)];
            # iterating over all mini batches
            mini_batch_x = X[:, batch_indexes]
            mini_batch_c = C[:, batch_indexes]

            # iterate over the mini_batch [previous_index, ... next_index]

            # grad = 0
            # for l in range(batch_size):
            #     grad += softmax_gradient_single(mini_batch_X[:,l], W, mini_batch_C[:,l])
            # grad = (1/batch_size)*grad

            grad = softmax_gradient(mini_batch_x, W, mini_batch_c)

            W = W - learning_rate * grad

        train_success_rate, validation_success_rate = check_predication(W, X, x_valid, c_training, c_validation)
        history.append([train_success_rate, validation_success_rate])

        if i % 100 == 0:
            print('loss: ', softmax_objective(X, W, C), ' epoch: ', i)

            print("train success rate is: " + str(train_success_rate * 100) + "%" + "  validation success rate is: "
                  + str(validation_success_rate * 100) + "%")
            # appending data for the plots
            train_rate_data.append(train_success_rate * 100)
            validation_rate_data.append(validation_success_rate * 100)
            epoch_data.append(i)

    return history, W, train_success_rate, validation_success_rate, epoch_data, train_rate_data, validation_rate_data


def check_predication(W, X, Xvalid, c_training, c_validation, num_of_samples=1000):
    training_idx = np.random.randint(0, X.shape[1] - 1, num_of_samples)
    validation_idx = np.random.randint(0, Xvalid.shape[1] - 1, num_of_samples)
    training_pred = predict(W, X[:, training_idx])
    # for debugging
    # check_train = Ctraining[training_idx]
    train_errors = training_pred - c_training[training_idx]

    validation_pred = predict(W, Xvalid[:, validation_idx])
    # for debugging
    # check_valid = c_validation[validation_idx]
    validation_errors = (validation_pred - c_validation[validation_idx])
    train_success = sum(train_errors == 0) / num_of_samples
    validation_success = sum(validation_errors == 0) / num_of_samples
    return train_success, validation_success


# uses the weights W in order to predict the values of X
def predict(W, X):
    prob = np.matmul(X.transpose(), W)
    res = prob.argmax(axis=1)
    return res


# calculates the value of ReLU(X)
def ReLU(X):
    X[X < 0] = 0
    return X


# builds the layers of a network (weights and biases) according to specified layer seizes.
# the first and last layer sizes are built according to the dimension of the data and the
# number of classes.
def build_layers(data_dimension, num_of_classes, layer_sizes):
    # data structures for the weights and biases
    W = []
    B = []

    last_dim = data_dimension
    num_of_layers = len(layer_sizes)

    # building hidden layers
    for i in range(num_of_layers):
        W_i = np.zeros((layer_sizes[i], last_dim))
        b_i = np.zeros(layer_sizes[i])
        last_dim = layer_sizes[i]
        W.append(W_i)
        B.append(b_i)

    # adding the last layer
    W_i = np.zeros((num_of_classes, last_dim))
    W.append(W_i)

    return np.asarray(W), np.asarray(B)


# calculates the forward propagation of the NN and returns the current loss
# as well as the ReLU derivatives of each hidden layer (those that are needed for
# the backward propagation)
def forward_propagation(W, X, B, C):
    relu_derivatives = []
    x_history = []
    x_i = X

    for i in range(B.shape[0]-1):
        x_history.append(x_i)
        x_i = ReLU(np.matmul(W[i], x_i) + B[i])
        relu_derivatives.append(x_i > 0)

    return softmax_objective(X, W, C), relu_derivatives, x_history


# going through each layer and preforming the gradient descent on the biases
# and the weights.
def backward_propagation(W, X, B, C, relu_derivative, x_history, learning_rate):

    # last layer gradient decent
    grad = softmax_gradient(X, W[-1], C)
    W[-1] = W[-1] - learning_rate * grad

    x_grad = softmax_data_gradient(X, W, C)

    # going through all hidden layers
    for i in range(B.shape[0] - 1, -1, -1):
        B[i] = B[i] - learning_rate * relu_derivative[i]
        x_grad = x_grad
        W[i] = W[i] - learning_rate * grad

    return W, B


def NN_SGD(X, C, layer_sizes, max_iter=50, learning_rate=0.02, batch_size=1000):
    W, B = build_layers(X.shape[0], C.shape[0], layer_sizes)

    num_of_mini_batches = round(X.shape[1] / batch_size)
    perm = np.random.permutation(X.shape[1])

    for i in range(max_iter):
        num_of_mini_batches = round(X.shape[1] / batch_size)
        perm = np.random.permutation(X.shape[1])

        # chose learning rate to advance thusly
        learning_rate = 1 / np.sqrt(i + 1)

        for j in range(num_of_mini_batches):
            batch_indexes = perm[(j * batch_size):((j + 1) * batch_size)];
            # iterating over all mini batches
            mini_batch_X = X[:, batch_indexes]
            mini_batch_C = C[:, batch_indexes]

            loss, relu_derivatives, x_history = forward_propagation(mini_batch_X, W, mini_batch_C, B)

            W, B = backward_propagation(mini_batch_X, W, mini_batch_C, B, relu_derivatives, x_history,  learning_rate)


# returns the gradient of the layer with respect to b as a matrix (why not as a vector?)
def ReLU_Gradient_by_b(W, X, b):
    return np.diag((np.matmul(W, X) + b) > 0)


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
    print(softmax_objective(A, W, C))
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


def running_on_mnist_data_set():
    # reading the training data
    Y = mnist_handler.read_label_file(
        'mnist/train-labels.idx1-ubyte')
    X = mnist_handler.read_image_file(
        'mnist/train-images.idx3-ubyte')
    # reading the test data
    Ytest = mnist_handler.read_label_file(
        'mnist/t10k-labels.idx1-ubyte')
    Xtest = mnist_handler.read_image_file(
        'mnist/t10k-images.idx3-ubyte')

    # transforming the matrices representing the images into vectors of pixels
    Xtest.shape = (np.shape(Xtest)[0], np.shape(Xtest)[1] * np.shape(Xtest)[2])

    Xtest = Xtest / 255

    bias_row = np.ones(Xtest.shape[0])
    Xtest = Xtest.transpose()
    Xtest = np.vstack([Xtest, bias_row])

    print(X.shape)
    print(Y.shape)

    C = np.zeros((10, Y.shape[0]))

    # creating C as required
    for i in range(10):
        C[i, :] = Y == i

    print(C.shape)

    c_test = np.zeros((10, Ytest.shape[0]))

    for i in range(10):
        c_test[i, :] = Ytest == i


    X.shape = (X.shape[0], X.shape[1] * X.shape[2])

    print(X.shape)
    print(Xtest.shape)

    X1 = X.transpose()

    X = X1 * (1 / 255)

    m = X.shape[1]
    n = X.shape[0]
    l = C.shape[0]

    bias_row = np.ones(m)

    X = np.vstack([X, bias_row])

    W = np.zeros((n + 1, l))

    history, W,  train_success_rate, validation_success_rate, epoch_data, train_rate_data, validation_rate_data\
        = stochastic_gradient_descent(X, W, C, is_mnist_data=True, x_valid=Xtest, c_valid=c_test)

    plot_results(epoch_data,train_rate_data, validation_rate_data, "MNIST")
    res = predict(W, X)

    print(sum(res - Y != 0))
    print("the number of labeled train data: " + str(Y.shape[0]))

    res = predict(W, Xtest)

    print(sum(res - Ytest != 0))
    print("the number of labeled validation data: " + str(Y.shape[0]))


def load_data_set(data_set_name):
    data = scipy.io.loadmat(data_set_name)
    ct = data['Ct']
    cv = data['Cv']
    yt = data['Yt']
    yv = data['Yv']
    return ct, cv, yt, yv


def plot_results(epoch_data, train_rate_data, validation_rate_data, example_data):
    plt.plot(epoch_data, train_rate_data, label="train success rate")
    plt.plot(epoch_data, validation_rate_data, label="validation success rate")
    # naming the x axis
    plt.xlabel('num of epochs')
    # naming the y axis
    plt.ylabel('success rate %')
    # giving a title to my graph
    plt.title('softmax classification for ' + example_data + " data set")
    plt.legend()
    # function to show the plot
    plt.show()


if __name__ == "__main__":
    # ================================================================================================
    # ================================================================================================
    # =============================          Eran's Data        ======================================
    # ================================================================================================
    # ================================================================================================
    Gmm = 'GMMData.mat'
    Peaks = 'PeaksData.mat'
    SwissRoll = 'SwissRollData.mat'

    example_data = Gmm
    Ct, Cv, Yt, Yv = load_data_set(example_data)
    # adding bias
    m = Yt.shape[1]
    bias_row = np.ones(m)
    Yt = np.vstack([Yt, bias_row])

    m = Yv.shape[1]
    bias_row = np.ones(m)
    Yv = np.vstack([Yv, bias_row])

    n = Yt.shape[0]
    l = Ct.shape[0]

    W = np.ones((n, l))
    learning_rate = 0.1
    batch_size = 10_000
    history, W, train_success_rate, validation_success_rate, epoch_data, train_rate_data, validation_rate_data\
        = stochastic_gradient_descent(Yt, W, Ct, Yv, Cv, max_iter=10_000)
    print("after running SGD on: " + example_data + " train success rate is: " + str(train_success_rate * 100) + "%" + "  validation success rate is: "
          + str(validation_success_rate * 100) + "%" + " learning rate is : " + str(learning_rate)
          + " mini_batch size is " + str(batch_size))

    plot_results()
# ================================================================================================
# ================================================================================================
# =============================             MNIST           ======================================
# ================================================================================================
# ================================================================================================

    running_on_mnist_data_set()
# ================================================================================================
# ================================================================================================
# =============================       Gradient testing      ======================================
# ================================================================================================
# ================================================================================================

# gradient_test_by_w()
#  TODO: refactor it so wil not be duplicate code
# gradient_test_x()
