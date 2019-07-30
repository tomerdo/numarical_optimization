import scipy.io
import numpy as np
import mnist_handler
import matplotlib.pyplot as plt
import stochastic_gradient_descent as sgd
import gradients as grads
import gradient_test as grad_test


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
        layer_size_i = layer_sizes[i]
        W_i = np.random.randn(layer_size_i, last_dim)  # connecting the layers through the shapes
        b_i = np.random.randn(layer_size_i)
        last_dim = layer_size_i
        W.append(W_i)
        B.append(b_i)

    # adding the last layer, +1 is for biases
    w_last_layer = np.random.randn(num_of_classes, last_dim + 1)
    W.append(w_last_layer)
    W = np.asarray(W)

    return W, np.asarray(B)


# calculates the forward propagation of the NN and returns the current loss
# as well as the ReLU derivatives of each hidden layer (those that are needed for
# the backward propagation)
def forward_propagation(W, X, B):
    relu_derivatives = []
    x_history = []
    x_i = X
    x_history.append(x_i)
    for i in range(B.shape[0]):
        mul_res = np.matmul(W[i], x_i)
        x_i = ReLU(mul_res + grads.pump(B[i], mul_res.shape[0], mul_res.shape[1]))  # activation function
        x_history.append(x_i)  # used later for back propagation
        relu_derivatives.append(x_i > 0)

    return relu_derivatives, x_history


# going through each layer and preforming the gradient descent on the biases
# and the weights.
def backward_propagation(W, X, B, C, relu_derivative, x_history, learning_rate):
    # last layer gradient decent
    grad = grads.softmax_gradient(x_history[-1], np.transpose(W[-1]), C)
    W[-1] = W[-1] - learning_rate * np.transpose(grad)

    temp_w = W[-1]
    # loss function gradient w.r.t X , excluding the last row of W ( the biases row)
    x_grad = grads.softmax_data_gradient(x_history[-1], np.transpose(temp_w[:, :-1]), C)

    # going through all hidden layers
    for i in range(B.shape[0] - 1, -1, -1):
        B[i] = B[i] - learning_rate * grads.JacV_b(relu_derivative[i], x_grad)  # updating B by the jacobian of B
        # updating W by the jacobian of W
        W[i] = W[i] - learning_rate * grads.JacV_w(x_history[i], relu_derivative[i], x_grad)
        x_grad = grads.JacV_x(W[i], relu_derivative[i], x_grad)

    return W, B


# this function implements stochastic gradient descent on neural networks ,
# it computes forward and backward propogation on the mini batches
# and updating the value of train_rate_data=[], validation_rate_data=[], epoch_data=[]
# for plotting reasons
def nn_sgd(X, C, layer_sizes, max_iter=50, x_valid=None, c_valid=None, learning_rate=0.1, batch_size=100,
           train_rate_data=[], validation_rate_data=[], epoch_data=[]):
    W, B = build_layers(X.shape[0], C.shape[0], layer_sizes)
    c_training, c_validation = sgd.rearrange_labels(C, c_valid)
    x_loss = np.zeros((layer_sizes[-1], X.shape[1]))

    for i in range(max_iter):
        num_of_mini_batches = round(X.shape[1] / batch_size)
        perm = np.random.permutation(X.shape[1])

        # chose learning rate to advance thusly
        learning_rate = 0.01
        # 1 / (i + 1)
        # learning_rate = 0.01

        for j in range(num_of_mini_batches):
            batch_indexes = perm[(j * batch_size):((j + 1) * batch_size)]
            # iterating over all mini batches
            mini_batch_x = X[:, batch_indexes]
            mini_batch_c = C[:, batch_indexes]

            relu_derivatives, x_history = forward_propagation(W, mini_batch_x, B)

            W, B = backward_propagation(W, mini_batch_x, B, mini_batch_c, relu_derivatives, x_history, learning_rate)

            if i % 100 == 0:
                x_loss[:, batch_indexes] = x_history[-1]

        if i % 100 == 0:
            # results are affected only by the last W layer
            print('loss: ', sgd.softmax_objective(x_loss, np.transpose(W[-1]), C), ' epoch: ', i)
            train_success_rate, validation_success_rate = check_predication(W, X, B, x_valid, c_training, c_validation)
            print("train success rate is: " + str(train_success_rate * 100) + "%" + "  validation success rate is: "
                  + str(validation_success_rate * 100) + "%")
            train_rate_data.append(train_success_rate * 100)
            validation_rate_data.append(validation_success_rate * 100)
            epoch_data.append(i)

    return train_rate_data, validation_rate_data, epoch_data


# handling the mnist data
# before running sgd
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
    # Xtest = np.vstack([Xtest, bias_row])

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

    # X = np.vstack([X, bias_row])

    W = np.zeros((n + 1, l))

    train_success_rate, validation_success_rate, epoch_data = nn_sgd(X, C, [3, 4, 6], x_valid=Xtest, c_valid=c_test)

    history, W, train_success_rate, validation_success_rate, epoch_data, train_rate_data, validation_rate_data \
        = sgd.stochastic_gradient_descent(X, W, C, x_valid=Xtest, c_valid=c_test)

    plot_results(epoch_data, train_rate_data, validation_rate_data, "MNIST")
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


# this function is to plot the data of the learning and the classification
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


# print to stdout result after running sgd learning
def print_result(example_data, train_success_rate, validation_success_rate, learning_rate, batch_size):
    print("after running SGD on: " + example_data + " train success rate is: " + str(
        train_success_rate * 100) + "%" + "  validation success rate is: "
          + str(validation_success_rate * 100) + "%" + " learning rate is : " + str(learning_rate)
          + " mini_batch size is " + str(batch_size))


# given W , B (computed by the SGD)
# run forward propagation on X
# and result the predicted label
# used for plotting the accuracy of the learning
def predict(W, X, B):
    x_i = X
    for i in range(B.shape[0]): # running forward pass
        mul_res = np.matmul(W[i], x_i)
        x_i = ReLU(mul_res + grads.pump(B[i], mul_res.shape[0], mul_res.shape[1]))
    prob = sgd.softmax(x_i, np.transpose(W[-1]))
    res = prob.argmax(axis=1)
    return res


# return the accuracy of the prediction on  <num_of_samples> samples
# calculated both training accuracy and validation accuracy
def check_predication(W, X, B, Xvalid, c_training, c_validation, num_of_samples=1000):
    training_idx = np.random.randint(0, X.shape[1] - 1, num_of_samples)
    validation_idx = np.random.randint(0, Xvalid.shape[1] - 1, num_of_samples)
    training_pred = predict(W, X[:, training_idx], B)
    # for debugging
    # check_train = Ctraining[training_idx]
    train_errors = training_pred - c_training[training_idx]

    validation_pred = predict(W, Xvalid[:, validation_idx], B)
    # for debugging
    # check_valid = c_validation[validation_idx]
    validation_errors = (validation_pred - c_validation[validation_idx])
    train_success = sum(train_errors == 0) / num_of_samples
    validation_success = sum(validation_errors == 0) / num_of_samples
    return train_success, validation_success

# the driver of this all project
# you can comment and uncomment relecent code snippets to run (according the question you want to check)
# in default it run nn_sgd on the GMM data with [15 , 5 ,10] layers
if __name__ == "__main__":
    # ================================================================================================
    # ================================================================================================
    # =============================          Eran's Data        ======================================
    # ================================================================================================
    # ================================================================================================
    Gmm = 'GMMData.mat'
    Peaks = 'PeaksData.mat'
    SwissRoll = 'SwissRollData.mat'

    example_data = SwissRoll
    Ct, Cv, Yt, Yv = load_data_set(example_data)  # loading data set

    m = Yt.shape[1]
    n = Yt.shape[0]
    l = Ct.shape[0]

    # ================================================================================================
    # ================================================================================================
    # ==================      (questions 1 - 3) simple softmax  with out NN         ==================
    # ================================================================================================
    # ================================================================================================

    # W = np.ones((n+1, l))
    # learning_rate = 0.1
    # batch_size = 10000
    # history, W, train_success_rate, validation_success_rate, epoch_data, train_rate_data, validation_rate_data\
    #     = sgd.stochastic_gradient_descent(Yt, W, Ct, Yv, Cv, max_iter=1_000)
    #
    # print_result(example_data, train_success_rate, validation_success_rate, learning_rate, batch_size)
    # plot_results(epoch_data, train_rate_data, validation_rate_data, example_data)

    # ================================================================================================
    # ================================================================================================
    # =============================             MNIST           ======================================
    # ================================================================================================
    # ================================================================================================

    # running_on_mnist_data_set()
    # ================================================================================================
    # ================================================================================================
    # =============================       Gradient testing      ======================================
    # ================================================================================================
    # ================================================================================================

    # grad_test.gradient_test_w(5,5,4)
    # gradient_test_x()
    #     grad_test.test_jac_b_t_v(5,7,3)
    #     grad_test.test_jac_x_t_v(5,5,3)
    #     grad_test.test_jac_w_t_v(5,5,5)
    # ================================================================================================
    # ================================================================================================
    # =============================       RUNNING NN (question 4 - 7)      ===========================
    # ================================================================================================
    # ================================================================================================
    train_rate_data, validation_rate_data, epoch_data = nn_sgd(Yt, Ct, layer_sizes=[10, 10, 10],
                                                               max_iter=10_000, x_valid=Yv, c_valid=Cv)
    plot_results(epoch_data, train_rate_data, validation_rate_data, "NN " + example_data)
    # running_on_mnist_data_set()
