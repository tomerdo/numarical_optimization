import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot


# The objective function for LASSO regression
def objective_function(x1, x2, mu):
    max1 = max(0, x1 ** 2 + x2 ** 2 - 5)
    max2 = max(0, -x1)
    return (x1+x2)**2 - 10*(x1+x2) + mu*(3*x1+x2-6)**2 + mu*max1**2 + mu*max2**2


# The gradient
def grad(x1, x2, mu):
    max1 = 0
    max2 = 0
    if max(0, x1 ** 2 + x2 ** 2 - 5) > 0:
        max1 = 4*x1*(x1**2 + x2**2 - 5)
    if max(0, -x1) > 0:
        max2 = 2*x1
    grad = np.ndarray(2)
    grad[0] = 2*x1 + 2*x2 - 10 +mu*6*(3*x1+x2-6) +mu*max1 + mu*max2
    grad[1] = 2*x1 + 2*x2 - 10 +mu*2*(3*x1+x2-6) +mu*max1

    return grad


def line_search(x, mu, grad, fx, alpha0=1, beta=0.5, c=0.0001, iterations=30):

    alphaj = alpha0
    flag = True
    iter = 0

    while flag:
        res = objective_function(*(x - alphaj*grad), mu) - fx - c * alphaj * sum(-grad * grad)
        iter += 1
        if res <= 0 or iter >= iterations:
            flag = False
        alphaj *= beta

    return alphaj


# Steepest descent method for LASSO regression
def SD(mu, x=None, iterations=10, tolerance=0.1, criteria=0):
    if x is None:
        x = np.zeros(2)

    # for plotting purposes
    plot = []

    for i in range(iterations):
        fx = objective_function(x[0], x[1], mu)
        plot.append(fx)

        gradf = grad(x[0], x[1], mu)

        # Calculating step size
        alpha = line_search(x, mu, gradf, fx)

        x = x - alpha * gradf
    return x, plot

if __name__ == "__main__":

    mus = [0.01, 0.1, 1, 10, 100]
    for mu in mus:

        x = np.ndarray(2)

        x[0] = 0
        x[1] = 0

        x, plot = SD(mu, x)

        print("mu is: " + str(mu) + " x is: " + str(x))

        pyplot.plot(plot, label=str(mu))

    pyplot.legend(title='mu values:')
    pyplot.show()


    # # gradient test
    #
    # # generating a random matrix and a random w
    # x = np.random.rand(2)
    # mu = 14
    #
    # # generating a random d
    # d = np.random.rand(2)
    # epsilon = 0.2
    #
    # # making the names shorter for convenience
    # f = lambda x: objective_function(x[0], x[1], mu)
    # gradf = lambda x: grad(x[0], x[1], mu)
    #
    # last1 = 1
    # last2 = 1
    #
    # # finding the ratio between the results in order to check correctness
    # # gradient test
    # for i in range(10):
    #     epsilon = (0.5)**i
    #
    #     curr1 = abs(f(x + epsilon * d)-f(x))
    #     ratio1 = curr1/last1
    #
    #     curr2 = abs(f(x + epsilon * d) - f(x) - epsilon * np.matmul(d, gradf(x)))
    #     ratio2 = curr2 / last2
    #
    #     last1 = curr1
    #     last2 = curr2
    #
    #     print(ratio1)
    #     print(ratio2)
    #     print('\n')