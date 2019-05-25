import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
from matplotlib import pyplot
import argparse


# Make the matrix A diagonally dominant by switching rows or die trying
# Returns:
# B - A possibly diagonally dominant matrix
# perm - a permutation of which row ended up where
# restore - the inverse of perm that satisfies A = B[restore]
def __diagonally_dominant__(A):
    if not isinstance(A, np.ndarray):
        print('Parameter is not ndarray')
        return
    n = (A.shape)[0]
    B = np.array(A)
    perm = np.array(range(n))
    for curr in range(n):
        max_idx = abs(B[:, curr]).argmax()
        B[[curr, max_idx]] = B[[max_idx, curr]]
        perm[[curr, max_idx]] = perm[[max_idx, curr]]
    restore = np.argsort(perm)
    return B, perm, restore


def Jacobi(A, b, x=None, iterations=100, tolerance=0.1, criteria=0, w=1):
    Add, perm, restore = __diagonally_dominant__(A)
    bdd = b[perm]
    if x is None:
        n = (A.shape)[0]
        x = np.zeros(n)

    diag = A.diagonal()
    D = np.diag(diag)
    invM = inv(D)

    first_plot = []
    second_plot = []

    if not criteria:
        for i in range(iterations):
            first_plot.append(norm(np.matmul(Add, x) - b))
            x_last = x
            x = x + w * np.matmul(invM, (bdd - np.matmul(Add, x)))
            second_plot.append(norm(np.matmul(Add, x) - b) / norm(np.matmul(Add, x_last) - b))
    else:
        flag = True
        counter = 0
        while flag:
            x_last = x
            x = x + w * np.matmul(invM, (bdd - np.matmul(Add, x)))
            counter += 1
            flag = (norm(x_last - x) > tolerance) and (counter <= iterations)

    return x, first_plot, second_plot


def Gauss_Seidel(A, b, x=None, iterations=100, tolerance=0.1, criteria=0, w=1):
    n = (A.shape)[0]
    Add, perm, restore = __diagonally_dominant__(A)
    bdd = b[perm]
    if x is None:
        x = np.zeros(n)

    LD = np.tril(A, 0)
    invM = inv(LD)

    first_plot = []
    second_plot = []

    if not criteria:
        for i in range(iterations):
            first_plot.append(norm(np.matmul(Add, x)-bdd))
            x_last = x
            x = x + w * np.matmul(invM, (bdd - np.matmul(Add, x)))
            second_plot.append(norm(np.matmul(Add, x) - bdd) / norm(np.matmul(Add, x_last) - bdd))

    else:
        flag = True
        counter = 0
        while flag:
            x_last = x
            x = x + w * np.matmul(invM, (bdd - np.matmul(Add, x)))
            counter += 1
            flag = (norm(x_last - x) > tolerance) and (counter <= iterations)

    return x, first_plot, second_plot


def __GD_iterate__(A,b,x,r):
    Ar = np.matmul(A, r)
    alpha = sum(r * r) / sum(r * Ar)
    r_new = r - alpha * Ar
    return x + alpha * r, r_new


def GD(A, b, x=None, iterations=100, tolerance=0.1, criteria=0):
    Add, perm, restore = __diagonally_dominant__(A)
    bdd = b[perm]
    if x is None:
        n = (A.shape)[0]
        x = np.zeros(n)

    first_plot = []
    second_plot = []

    r = bdd - np.matmul(Add, x)

    if not criteria:
        for i in range(iterations):
            first_plot.append(norm(np.matmul(Add, x) - bdd))
            x_last = x
            x,r =__GD_iterate__(Add, bdd, x, r)
            second_plot.append(norm(np.matmul(Add, x) - bdd) / norm(np.matmul(Add, x_last) - bdd))
    else:
        flag = True
        counter = 0
        while flag:
            first_plot.append(norm(np.matmul(Add, x) - bdd))
            x_last = x
            x,r = __GD_iterate__(Add, bdd, x, r)
            second_plot.append(norm(np.matmul(Add, x) - bdd) / norm(np.matmul(Add, x_last) - bdd))
            counter += 1
            flag = (norm(x_last - x) > tolerance) and (counter <= iterations)

    return x, first_plot, second_plot


def __CG_iterate__(A,b,x,r,p):
    Ap = np.matmul(A, p)
    alpha = sum(r * r) / sum(p * Ap)
    r_new = r - alpha * Ap
    beta = sum(r_new * r_new) / sum(r * r)
    p_new = r_new + beta * p
    return x + alpha * p, r_new, p_new


def CG(A, b, x=None, iterations=100, tolerance=0.1, criteria=0):
    Add, perm, restore = __diagonally_dominant__(A)
    bdd = b[perm]
    if x is None:
        n = (A.shape)[0]
        x = np.zeros(n)

    first_plot = []
    second_plot = []

    r = bdd - np.matmul(Add, x)
    p = r

    if not criteria:
        for i in range(iterations):
            first_plot.append(norm(np.matmul(Add, x) - bdd))
            x_last = x
            x, r, p =__CG_iterate__(Add, bdd, x, r, p)
            second_plot.append(norm(np.matmul(Add, x) - bdd) / norm(np.matmul(Add, x_last) - bdd))
    else:
        flag = True
        counter = 0
        while flag:
            first_plot.append(norm(np.matmul(Add, x) - bdd))
            x_last = x
            x, r, p = __CG_iterate__(Add, bdd, x, r, p)
            second_plot.append(norm(np.matmul(Add, x) - bdd) / norm(np.matmul(Add, x_last) - bdd))
            counter += 1
            flag = (norm(x_last - x) > tolerance) and (counter <= iterations)

    return x, first_plot, second_plot


def __GMRES1_iterate__(A,b,x,r):
    Ar = np.matmul(A, r)
    alpha = sum(r * Ar) / sum(Ar * Ar)
    r_new = r - alpha * Ar
    return x + alpha * r, r_new


def GMRES1(A, b, x=None, iterations=100, tolerance=0.1, criteria=0):
    Add, perm, restore = __diagonally_dominant__(A)
    bdd = b[perm]
    if x is None:
        n = (A.shape)[0]
        x = np.zeros(n)

    first_plot = []
    second_plot = []

    r = bdd - np.matmul(Add, x)

    if not criteria:
        for i in range(iterations):
            first_plot.append(norm(np.matmul(Add, x) - bdd))
            x_last = x
            x, r = __GMRES1_iterate__(Add, bdd, x, r)
            second_plot.append(norm(np.matmul(Add, x) - bdd) / norm(np.matmul(Add, x_last) - bdd))
    else:
        flag = True
        counter = 0
        while flag:
            first_plot.append(norm(np.matmul(Add, x) - bdd))
            x_last = x
            x, r = __GMRES1_iterate__(Add, bdd, x, r)
            second_plot.append(norm(np.matmul(Add, x) - bdd) / norm(np.matmul(Add, x_last) - bdd))
            counter += 1
            flag = (norm(x_last - x) > tolerance) and (counter <= iterations)

    return x, first_plot, second_plot

if __name__ == '__main__':

    # for 1)b.
    # A = 2.1*np.diag(np.ones(100))-np.diag(np.ones(99),-1)-np.diag(np.ones(99),1)
    # b = np.random.rand(100,1)
    # sol, plot1, plot2 = CG(A, b, None, 100, 0.1, 0)
    #
    # p, = pyplot.semilogy(plot1)
    # pyplot.legend([p], ['||Ax-b||'])
    #
    # p, = pyplot.semilogy(plot2)
    # pyplot.legend([p],['convergence factor'])
    #
    # pyplot.xlabel('iteration')
    # pyplot.ylabel('value')
    # pyplot.title('CG')
    # pyplot.show()
    # print(sol)

    # for 3)c.
    # A = np.array([[5, 4, 4, -1, 0],\
    #               [3, 12, 4, -5, -5],\
    #               [-4, 2, 6, 0, 3],\
    #               [4, 5, -7, 10, 2],\
    #               [1, 2, 5, 3, 10]])
    # b = np.array([1, 1, 1, 1, 1])
    # sol, plot1, plot2 = GMRES1(A, b, None, 50)
    # p, = pyplot.semilogy(plot1)
    # pyplot.legend([p], ['||Ax-b||'])
    # pyplot.xlabel('iteration')
    # pyplot.ylabel('value')
    # pyplot.title('GMERS1')
    # pyplot.show()
    # print(sol)
