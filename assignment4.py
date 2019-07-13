import scipy.io
import numpy as np
import mnist_handler
import matplotlib.pyplot as plt
import stochastic_gradient_descent as sgd


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

    return sgd.softmax_objective(X, W, C), relu_derivatives, x_history


# going through each layer and preforming the gradient descent on the biases
# and the weights.
def backward_propagation(W, X, B, C, relu_derivative, x_history, learning_rate):

    # last layer gradient decent
    grad = sgd.softmax_gradient(X, W[-1], C)
    W[-1] = W[-1] - learning_rate * grad

    x_grad = grad.softmax_data_gradient(X, W, C)

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
        = sgd.stochastic_gradient_descent(X, W, C, is_mnist_data=True, x_valid=Xtest, c_valid=c_test)

    plot_results(epoch_data,train_rate_data, validation_rate_data, "MNIST")
    res = sgd.predict(W, X)

    print(sum(res - Y != 0))
    print("the number of labeled train data: " + str(Y.shape[0]))

    res = sgd.predict(W, Xtest)

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
        = sgd.stochastic_gradient_descent(Yt, W, Ct, Yv, Cv, max_iter=10_000)
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
