import scipy.io
import numpy as np
import mnist_handler
import matplotlib.pyplot as plt
import gradients as grads


def softmax_objective(X, W, C):
    m = X.shape[1]
    l = W.shape[1]

    Xt = X.transpose()
    XtW = np.matmul(Xt, W)

    eta = XtW.max(axis=1)

    # pumping into matrix with shape (m,l)
    eta = grads.pump(eta, m, l)

    num = np.exp(XtW - eta)

    # summing rows to get denominator
    den = num.sum(axis=1)

    # pumping into matrix with shape (m,l)
    den = grads.pump(den, m, l)

    logDXtW = np.log(num / den)

    res = 0

    for i in range(l):
        res += sum(C[i, :] * logDXtW[:, i])

    return res * (-1 / m)


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

    training_labels = grads.pump(labels, C.shape[0], C.shape[1])
    c_training = (C * training_labels).sum(axis=0)

    validation_labels = grads.pump(labels, c_valid.shape[0], c_valid.shape[1])
    c_validation = (c_valid * validation_labels).sum(axis=0)

    for i in range(max_iter):
        num_of_mini_batches = round(X.shape[1] / batch_size)
        perm = np.random.permutation(X.shape[1])

        learning_rate = 1 / np.sqrt(i + 1)

        for j in range(num_of_mini_batches):
            batch_indexes = perm[(j * batch_size):((j + 1) * batch_size)]
            # iterating over all mini batches
            mini_batch_x = X[:, batch_indexes]
            mini_batch_c = C[:, batch_indexes]

            # iterate over the mini_batch [previous_index, ... next_index]

            # grad = 0
            # for l in range(batch_size):
            #     grad += softmax_gradient_single(mini_batch_X[:,l], W, mini_batch_C[:,l])
            # grad = (1/batch_size)*grad

            grad = grads.softmax_gradient(mini_batch_x, W, mini_batch_c)

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
