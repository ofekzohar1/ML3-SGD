#################################
# Your name: Ofek Zohar
#################################

import numpy as np
import numpy.random
from scipy.special import expit
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    size, n = data.shape
    w_curr = np.zeros(n)

    for t in range(1, T + 1):
        i = np.random.randint(size)  # Choose sample uniformly
        eta_t = eta_0 / t
        w_next = (1 - eta_t) * w_curr

        prod = labels[i] * np.dot(data[i], w_curr)  # yi * <w,xi>
        if prod < 1:
            w_next += eta_t * C * labels[i] * data[i]
        w_curr = w_next  # w_t+1
    return w_curr


def SGD_log(data, labels, eta_0, T):
    size, n = data.shape
    w = np.zeros(n)

    ws = []  # Save all w_t's
    for t in range(1, T + 1):
        ws.append(w)
        i = np.random.randint(size)  # Choose sample uniformly - equivalent to fi
        w = w - (eta_0 / t) * gradient(w, data[i], labels[i])  # w_t+1 update
    return w, ws  # w_T+1


# ############ Questions ################ #

def best_eta_p1(train_x, train_y, valid_x, valid_y, C, T):
    """
    Calc the best eta_0 using 2 sessions of cross validation with various resolutions. hinge loss.
    """
    # Low resolution
    etas = np.float_power(10, np.arange(-5, 4, 1))
    eta = cross_validation_p1(train_x, train_y, valid_x, valid_y, etas=etas, Cs=[C], T=T, scale="log", eta_or_C="eta")

    # High resolution
    eta_log = np.log10(eta)
    start, stop, step = eta_log - 1, eta_log + 1, 0.1
    etas = np.float_power(10, np.arange(start, stop, step))
    eta = cross_validation_p1(train_x, train_y, valid_x, valid_y, etas=etas, Cs=[C], T=T, scale="linear",
                              eta_or_C="eta")
    return eta


def best_C(train_x, train_y, valid_x, valid_y, eta, T):
    """
    Calc the best C using 2 sessions of cross validation with various resolutions
    """
    # Low resolution
    Cs = np.float_power(10, np.arange(-5, 4, 1))
    C = cross_validation_p1(train_x, train_y, valid_x, valid_y, etas=[eta], Cs=Cs, T=T, scale="log", eta_or_C="C")

    # High resolution
    C_log = np.log10(C)
    start, stop, step = C_log - 1, C_log + 1, 0.1
    Cs = np.float_power(10, np.arange(start, stop, step))
    C = cross_validation_p1(train_x, train_y, valid_x, valid_y, etas=[eta], Cs=Cs, T=T, scale="linear", eta_or_C="C")
    return C


def image_from_w(w):
    """
    Plot heat map (or weight) for w
    """
    plt.imshow(np.reshape(w, (28, 28)), interpolation="nearest")
    plt.show()


def best_eta_p2(train_x, train_y, valid_x, valid_y, T):
    """
    Calc the best eta_0 using 2 sessions of cross validation with various resolutions. log_loss.
    """
    # Low resolution
    etas = np.float_power(10, np.arange(-5, 6, 1))
    eta = cross_validation_p2(train_x, train_y, valid_x, valid_y, etas, T, "log")

    # High resolution
    eta_log = np.log10(eta)
    start, stop, step = eta_log - 1, eta_log + 1, 0.1
    etas = np.float_power(10, np.arange(start, stop, step))
    eta = cross_validation_p2(train_x, train_y, valid_x, valid_y, etas, T, "linear")
    return eta


# ############ helpers & main ################ #

def accuracy(data, labels, w):
    """
    Calculate the accuracy of the given classifier w on the training set S=(data,labels)
    """
    ctr = 0
    size = len(data)
    for i in range(size):
        sign = np.sign(np.dot(data[i], w))
        sign = sign if sign != 0 else 1
        ctr += 1 if sign != labels[i] else 0
    return 1 - ctr / size


def cross_validation_p1(train_data, train_labels, validation_data, validation_labels, etas, Cs, T, scale, eta_or_C):
    """
    Cross validation for the hinge loss. Usefully for finding best C and best eta_0.
    """
    acc_avgs = []
    for C in Cs:
        for eta in etas:
            avg_acc = 0
            for i in range(10):  # Average above 10 iterations
                w = SGD_hinge(train_data, train_labels, C, eta, T)
                avg_acc += accuracy(validation_data, validation_labels, w)
            avg_acc /= 10
            acc_avgs.append(avg_acc)

    plt.plot(etas if eta_or_C == "eta" else Cs, acc_avgs)
    plt.xlabel(eta_or_C)
    plt.xscale(scale)
    plt.ylabel("Average Accuracy")
    plt.show()

    argmax = np.argmax(acc_avgs)
    return etas[argmax] if eta_or_C == "eta" else Cs[argmax]


def cross_validation_p2(train_data, train_labels, validation_data, validation_labels, etas, T, scale):
    """
    Cross validation for the log loss
    """
    acc_avgs = []
    for eta in etas:
        avg_acc = 0
        for i in range(10):
            w, ws = SGD_log(train_data, train_labels, eta, T)
            avg_acc += accuracy(validation_data, validation_labels, w)
        avg_acc /= 10
        acc_avgs.append(avg_acc)

    plt.plot(etas, acc_avgs)
    plt.xlabel("eta")
    plt.xscale(scale)
    plt.ylabel("Average Accuracy")
    plt.show()

    return etas[np.argmax(acc_avgs)]


def gradient(w, data, label):
    """
    Calc the gradient of log_loss using the sigmoid function
    """
    grad = -1 * label * data  # -y * x
    prod = np.dot(w, data) * label  # <w,x> * y
    grad *= expit(prod)  # sigmoid(prod) * (-y*x) == gradient
    return grad


def plot_ws(ws):
    """
    Plot w's norm as function of the SGD iteration (t)
    """
    plt.plot(range(len(ws)), [np.linalg.norm(w) for w in ws])
    plt.xlabel("Iteration (t)")
    plt.xscale('log')
    plt.ylabel("Norm(w_t)")
    plt.show()


def main():
    train_x, train_y, valid_x, valid_y, test_x, test_y = helper()

    # Q5
    eta_0 = best_eta_p1(train_x, train_y, valid_x, valid_y, C=1, T=1000)
    print(eta_0, np.log10(eta_0))
    C = best_C(train_x, train_y, valid_x, valid_y, eta_0, T=1000)
    print(C, np.log10(C))
    w = SGD_hinge(np.concatenate((train_x, valid_x), axis=0), np.concatenate((train_y, valid_y), axis=0), C, eta_0,
                  T=20000)
    image_from_w(w)
    print(accuracy(test_x, test_y, w))

    # Q6
    eta_0 = best_eta_p2(train_x, train_y, valid_x, valid_y, T=1000)
    print(eta_0, np.log10(eta_0))
    w, ws = SGD_log(np.concatenate((train_x, valid_x), axis=0), np.concatenate((train_y, valid_y), axis=0), eta_0,
                    T=20000)
    image_from_w(w)
    print(accuracy(test_x, test_y, w))
    plot_ws(ws)


if __name__ == "__main__":
    main()
