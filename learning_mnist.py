import mnist_handler
import numpy as np
from matplotlib import pyplot

def mnist_GD(A, b, x=None, iterations=100, tolerance=0.1, criteria=0):

    if x is None:
        n = (A.shape)[0]
        x = np.zeros(n)


    for iter in range(iterations):
        x = x - alpha*grad



if __name__ == "__main__":
    Y = mnist_handler.read_label_file('D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-labels.idx1-ubyte')
    X = mnist_handler.read_image_file('D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-images.idx3-ubyte')

    X89, Y89 = mnist_handler.get_digits([8, 9], X, Y)

    #transforming the matrices representing the images into vectors of pixels
    X89.shape = (np.shape(X89)[0], np.shape(X89)[1]*np.shape(X89)[2])

    print(np.shape(X89))
    print(np.shape(Y89))

    x, plot1, plot2 = iterative_methods.GD(X89, Y89)

    pyplot.semilogy(plot1)
    pyplot.show()
