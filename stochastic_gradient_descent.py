import random
import numpy as np
# this class contains the implementation of SGD


def get_mini_batch_indexes(num_of_mini_batches, num_of_samples):
    list_of_indexes = random.sample(range(0, num_of_samples), num_of_mini_batches)
    list_of_indexes.sort()
    list_of_indexes.reverse()
    return list_of_indexes


def grad_of_softmax(samples, W):
    return 1


#  def stochastic_gradient_descent_non_equal_batches(W = [], samples = [] ,max_iter=100, learning_rate=0.01, num_of_mini_batches = 50):
#     for i in range(max_iter):
#         group_of_random_indexes = get_mini_batch_indexes(num_of_mini_batches, samples.shape[0])
#         next_index = 0
#
#         # iterating over all mini batches
#         while len(group_of_random_indexes) > 0:
#             previous_index = next_index
#             next_index = group_of_random_indexes.pop()
#             mini_batch = samples[previous_index:next_index]
#
#             # iterate over the mini_batch [previous_index, ... next_index]
#             grad = grad_of_softmax(mini_batch, W)
#             W = W + learning_rate * grad
#
#             # handling the last mini batch
#             if len(group_of_random_indexes) == 0:
#                 last_mini_batch = samples[next_index:]
#                 grad = grad_of_softmax(last_mini_batch, W)
#                 W = W + learning_rate * grad
#
#         return W

def stochastic_gradient_descent(W = [], samples = [] ,max_iter=100, learning_rate=0.01, batch_size = 50):

    for i in range(max_iter):
        num_of_mini_batches = (samples.shape[0] / batch_size) - 1
        perm = np.random.permutation(samples.shape[0])

        for j in range(num_of_mini_batches):
            batch_indexes = perm[(j * batch_size):((j + 1) * batch_size)];
            # iterating over all mini batches
            mini_batch = samples[batch_indexes]
            # iterate over the mini_batch [previous_index, ... next_index]
            grad = grad_of_softmax(mini_batch, W)
            W = W + learning_rate * grad
        return W
