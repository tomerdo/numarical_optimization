import numpy as np
import matplotlib.pyplot as plt


def read_label_file(path):
    file = open(path, "rb")

    #reading the magic number
    magic_number = int.from_bytes(file.read(4), byteorder='big')
    print('magic number is:')
    print(magic_number)

    # reading the size of the file
    num_of_labels = int.from_bytes(file.read(4), byteorder='big')
    print('number of labels is:')
    print(num_of_labels)

    # reading the labels and converting to np.array
    Y = np.frombuffer(file.read(), dtype=np.uint8, count=-1, offset=0)

    return Y

    file.close()


def read_image_file(path):
    file = open(path, "rb")

    #reading the magic number
    magic_number = int.from_bytes(file.read(4), byteorder='big')
    print('magic number is:')
    print(magic_number)

    # reading the size of the file
    num_of_images = int.from_bytes(file.read(4), byteorder='big')
    print('number of images is:')
    print(num_of_images)

    # reading the size of each image
    num_of_rows = int.from_bytes(file.read(4), byteorder='big')
    num_of_cols = int.from_bytes(file.read(4), byteorder='big')
    print('number of rows is:')
    print(num_of_rows)
    print('number of columns is:')
    print(num_of_cols)

    X = np.frombuffer(file.read(), dtype=np.uint8, count=-1, offset=0)
    X.shape = (num_of_images, num_of_rows, num_of_cols)

    return X

    file.close()


def get_digits(digits, images, labels):

    Y = []

    for y in labels:
        Y.append(y in digits)

    indices = np.nonzero(Y)

    X = images[indices]
    Y = labels[indices]

    return X, Y


if __name__ == "__main__":
    Y = read_label_file('D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-labels.idx1-ubyte')
    X = read_image_file('D:\\programs\\pycharm\projects\\numarical_optimization2\\mnist\\t10k-images.idx3-ubyte')

    X89, Y89 = get_digits([8, 9], X, Y)

    print(np.shape(X89))

    plt.imshow(X89[132], cmap='hot', interpolation='nearest')
    plt.show()
    # print(np.shape(X))
    # print(type(X))
