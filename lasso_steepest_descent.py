import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot


# The objective function for LASSO regression
def objective_function(A, u, v, b, lamda):
    return norm(np.matmul(A, u-v) - b)**2 + sum(lamda * u) + sum(lamda * v)


# The gradient by u
def gradu(A, u, v, b, lamda):
    AAu = np.matmul(np.transpose(A),np.matmul(A, u))
    AAv = np.matmul(np.transpose(A), np.matmul(A, v))
    Ab = np.matmul(np.transpose(A), b)
    return 2 * (AAu - AAv - Ab) + lamda


# The gradient by v
def gradv(A, u, v, b, lamda):
    AAu = np.matmul(np.transpose(A),np.matmul(A, u))
    AAv = np.matmul(np.transpose(A), np.matmul(A, v))
    Ab = np.matmul(np.transpose(A), b)
    return 2 * (AAv - AAu + Ab) + lamda


# Line search for v (same line search but we thought the code would be more readable this way)
def line_search_v(u, v, lamda, b, grad_v, fx, alpha0=1, beta=0.5, c=0.0001, iterations=100):

    alphaj = alpha0
    flag = True
    iter = 0

    while flag:
        new_v = v - alphaj*grad_v
        new_v = new_v * (new_v > 0)

        res_v = objective_function(A, u, new_v, b, lamda) - fx - c * alphaj * sum(-grad_v * grad_v)
        iter += 1
        if res_v <= 0 or iter >= iterations:
            flag = False
        alphaj *= beta

    return alphaj

# Linesearch for u

def line_search_u(u, v, lamda, b, grad_u, fx, alpha0=1, beta=0.5, c=0.0001, iterations=30):

    alphaj = alpha0
    flag = True
    iter = 0

    while flag:
        new_u = u - alphaj*grad_u
        new_u = new_u * (new_u > 0)

        res_u = objective_function(A, new_u, v, b, lamda) - fx - c * alphaj * sum(-grad_u * grad_u)
        iter += 1
        if res_u <= 0  or iter >= iterations:
            flag = False
        alphaj *= beta

    return alphaj


# Steepest descent method for LASSO regression
def SD(A, b, lamda, x=None, iterations=100, tolerance=0.1, criteria=0):
    if x is None:
        n = (A.shape)[1]
        x = np.zeros(n)

    # dividing x into positive an negative entries
    u = x * (x > 0)
    v = abs(x * (x < 0))

    # for plotting purposes
    plot = []

    for i in range(iterations):
        fx = objective_function(A, u, v, b, lamda)
        plot.append(fx)

        # We compute seperate steps for v and u using 2 different gradients (Jacobian actually)
        grad_u = gradu(A, u, v, b, lamda)
        grad_v = gradv(A, u, v, b, lamda)

        # Calculating step sizes for each descent
        alphau = line_search_u(u, v, lamda, b, grad_u, fx)
        alphav = line_search_v(u, v, lamda, b, grad_v, fx)

        u = u - alphau * grad_u
        v = v - alphav * grad_v

        # projecting u and v back into the feasible space
        u = u * (u > 0)
        v = v * (v > 0)

    return u, v, plot

if __name__ == "__main__":

    A = np.random.rand(100, 200)

    # setting up x as a sparse vector
    x = np.zeros(200)
    nonzero=[]
    for i in range(20):
        idx = np.random.randint(0, 199)
        while idx in nonzero:
            idx = np.random.randint(0, 199)
        nonzero.append(idx)
        x[idx] = np.random.randn(1)[0]

    # creating b using a normal noise addition
    eta = np.random.randn(100) * 0.1
    b = np.matmul(A,x) + eta

    #8.7
    lamda = 8

    u, v, plot = SD(A, b, lamda)

    print(len(np.nonzero(u - v)[0]))

    print(np.nonzero(u - v)[0])
    print(np.nonzero(x)[0])
    # print(u-v)

    # print(x)

    # print(x - (u-v))

    # print(np.matmul(A,x))

    pyplot.plot(plot)
    pyplot.show()

    # print(np.matmul(A,u-v)-b)

    # gradient test

    # # generating a random matrix and a random w
    # A = np.random.rand(8, 5)
    # u = np.random.rand(5)
    # v = np.random.rand(5)
    # lamda = np.random.rand(1)[0]
    # b = np.random.randn(8)
    # lamda = 3
    #
    # # generating a random d
    # d = np.random.rand(5)
    # epsilon = 0.2
    #
    # # making the names shorter for convenience
    # f = lambda v: objective_function(A, u, v, b, lamda)
    # gradf = lambda v: gradv(A, u, v, b, lamda)
    #
    # last1 = 1
    # last2 = 1
    #
    # # finding the ratio between the results in order to check correctness
    # # gradient test
    # for i in range(10):
    #     epsilon = (0.5)**i
    #
    #     new_v = v + epsilon * d
    #     new_v = new_v * (new_v>0)
    #
    #     curr1 = abs(f(new_v)-f(v))
    #     ratio1 = curr1/last1
    #
    #     curr2 = abs(f(new_v) - f(v) - epsilon * np.matmul(d, gradf(v)))
    #     ratio2 = curr2 / last2
    #
    #     last1 = curr1
    #     last2 = curr2
    #
    #     print(ratio1)
    #     print(ratio2)
    #     print('\n')

