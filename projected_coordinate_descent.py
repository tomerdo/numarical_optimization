import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
from matplotlib import pyplot


def projected_coordinate_descent(h, g, a, b, iterations=100):
    n = g.shape[0]
    x = np.zeros(n)
    for iter in range(iterations):
        for i in range(n):
            m_i = compute_m_i(x, i, h, g)
            print("m_i for index " + str(i) + " in iter " + str(iter) + " is " + str(m_i))
            if m_i < a[i]:
                x[i] = a[i]
            elif m_i > b[i]:
                x[i] = b[i]
            else:  # a_i <= m_i <= b_i
                x[i] = m_i

        print("in iteration: " + str(iter) + " x is : " + str(x))
    return x


def compute_m_i(x, i, h, g):
    x_comp = np.delete(x, i)
    h_i_comp = np.delete(h[i], i)

    nom = np.matmul(h_i_comp, x_comp) - g[i]
    return np.around(nom / h[i][i], 3)


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
    #
    # # for 3)c.
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

    # for ass3.q3.d

    h = np.array([[5, -1, -1, -1, -1],
                  [-1, 5, -1, -1, -1],
                  [-1, -1, 5, -1, -1],
                  [-1, -1, -1, 5, -1],
                  [-1, -1, -1, -1, 5],
                  ])
    a_val = 0
    b_val = 5
    g = np.array([18, 6, -12, -6, 18])
    a = np.array([a_val, a_val, a_val, a_val, a_val])
    b = np.array([b_val, b_val, b_val, b_val, b_val])
    projected_coordinate_descent(h, g, a, b)
