import numpy as np
import ctypes
import matplotlib.pyplot as plt

A = np.array([
    [1, 1],
    [2, 3],
    [3, 3]
])
B = np.array([
    1,
    -1,
    -1
])

C = np.concatenate(
    [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([0.5, 2])])
D = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

E = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
F = np.array([1, 1, -1, -1])

G = np.random.random((500, 2)) * 2.0 - 1.0
H = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in G])

I = np.random.random((500, 2)) * 2.0 - 1.0
J = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
              [0, 0, 0] for p in I])

I = I[[not np.all(arr == [0, 0, 0]) for arr in J]]
J = J[[not np.all(arr == [0, 0, 0]) for arr in J]]

K = np.random.random((1000, 2)) * 2.0 - 1.0
L = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
    p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in K])

M = np.array([
    [1],
    [2]
])
N = np.array([
    2,
    3
])

O = np.array([
    [1],
    [2],
    [3]
])
P = np.array([
    2,
    3,
    2.5
])

Q = np.array([
    [1, 1],
    [2, 2],
    [3, 1]
])
R = np.array([
    2,
    3,
    2.5
])

S = np.array([
    [1, 1],
    [2, 2],
    [3, 3]
])
T = np.array([
    1,
    2,
    3
])

U = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
])
V = np.array([
    2,
    1,
    -2,
    -1
])


def initW(points, labels, randW):
    W = np.random.uniform(-1, 1, 3)
    for _ in range(10000):
        k = np.random.randint(0, len(labels))
        Yk = labels[k]
        Xk = np.array([1, points[k][0], points[k][1]])
        signal = np.matmul(W, Xk)
        gXk = 1.0 if np.squeeze(signal) >= 0 else -1.0
        W = W + 0.01 * (Yk - gXk) * Xk

    return W


def initWMultiClass(points, labels, column, W):
    for _ in range(10000):
        k = np.random.randint(0, len(labels))
        if (labels[k][column] == 1):
            Yk = 1
        else:
            Yk = -1

        Xk = np.array([1, points[k][0], points[k][1]])
        signal = np.matmul(W, np.array(Xk))
        gXk = 1.0 if np.squeeze(signal) >= 0 else -1.0
        W = W + 0.01 * (Yk - gXk) * Xk

    return W

def linearModel(W):
    predicted_labels = []
    predicted_x1 = []
    predicted_x2 = []
    for x1 in range(0, 300):
        for x2 in range(0, 300):
            predicted_x1.append(x1 / 100)
            predicted_x2.append(x2 / 100)
            predicted_labels.append('lightskyblue' if x1 / 100 * W[1] + x2 / 100 * W[2] + W[0] >= 0 else 'pink')

    return predicted_x1, predicted_x2, predicted_labels

def linearModelMultiClass(W1, W2, W3):
    predicted_labels = []
    predicted_x1 = []
    predicted_x2 = []
    for x1 in range(-100, 100):
        for x2 in range(-100, 100):
            if (x1 / 100 * W1[1] + x2 / 100 * W1[2] + W1[0]) >= (x1 / 100 * W2[1] + x2 / 100 * W2[2] + W2[0]) and (x1 / 100 * W1[1] + x2 / 100 * W1[2] + W1[0]) >= (x1 / 100 * W3[1] + x2 / 100 * W3[2] + W3[0]):
                predicted_x1.append(x1 / 100)
                predicted_x2.append(x2 / 100)
                predicted_labels.append('lightskyblue')
            elif x1 / 100 * W2[1] + x2 / 100 * W2[2] + W2[0] >= x1 / 100 * W3[1] + x2 / 100 * W3[2] + W3[0]:
                predicted_x1.append(x1 / 100)
                predicted_x2.append(x2 / 100)
                predicted_labels.append('pink')
            else:
                predicted_x1.append(x1 / 100)
                predicted_x2.append(x2 / 100)
                predicted_labels.append('lightgreen')
    return predicted_x1, predicted_x2, predicted_labels

def regressionModel(X, Y):
    ones_column = np.ones((X.shape[0], 1))
    X_design = np.hstack((X, ones_column))
    coeffs = np.linalg.lstsq(X_design, Y, rcond=None)[0]
    slope = coeffs[0]
    intercept = coeffs[1]

    def linear_regression(x):
        return slope * x + intercept

    x_pred = np.linspace(np.min(X), np.max(X), 100)
    y_pred = linear_regression(x_pred)
    return x_pred, y_pred


def printScatter(predicted_x1, predicted_x2, predicted_labels, classeAX, classeAY, classeBX, classeBY):
    plt.scatter(predicted_x1, predicted_x2, c=predicted_labels)
    plt.scatter(classeAX, classeAY, color='blue')
    plt.scatter(classeBX, classeBY, color='red')
    plt.show()
    plt.clf()


def showValue():
    plt.show()
    plt.clf()


def executeTest(test):
    if test == 1:
        # Cas de test 1
        printScatter(*linearModel(initW(A, B, 3)), A[0, 0], A[0, 1], A[1:3, 0], A[1:3, 1])
    elif test == 2:
        # Cas de test 2
        printScatter(*linearModel(initW(C, D, 3)), C[0:50, 0], C[0:50, 1], C[50:100, 0], C[50:100, 1])
    elif test == 3:
        # Cas de test 3
        printScatter(*linearModel(initW(E, F, 3)), E[0:2, 0], E[0:2, 1], E[2:4, 0], E[2:4, 1])
    elif test == 4:
        # Cas de test 4
        predicted_x1, predicted_x2, predicted_labels = linearModel(initW(G, H, 3))
        plt.scatter(predicted_x1, predicted_x2, c=predicted_labels)
        plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: H[c[0]] == 1, enumerate(G)))))[:, 0],
                    np.array(list(map(lambda elt: elt[1], filter(lambda c: H[c[0]] == 1, enumerate(G)))))[:, 1],
                    color='blue')
        plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: H[c[0]] == -1, enumerate(G)))))[:, 0],
                    np.array(list(map(lambda elt: elt[1], filter(lambda c: H[c[0]] == -1, enumerate(G)))))[:, 1],
                    color='red')
        showValue()
    elif test == 5:
        # Cas de test 5
        W = np.random.uniform(-1, 1, 3)
        predicted_x1, predicted_x2, predicted_labels = linearModelMultiClass(initWMultiClass(I, J, 0, W), initWMultiClass(I, J, 1, W), initWMultiClass(I, J, 2, W))
        plt.scatter(predicted_x1, predicted_x2, c=predicted_labels)
        plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: J[c[0]][0] == 1, enumerate(I)))))[:, 0],
                    np.array(list(map(lambda elt: elt[1], filter(lambda c: J[c[0]][0] == 1, enumerate(I)))))[:, 1],
                    color='blue')
        plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: J[c[0]][1] == 1, enumerate(I)))))[:, 0],
                    np.array(list(map(lambda elt: elt[1], filter(lambda c: J[c[0]][1] == 1, enumerate(I)))))[:, 1],
                    color='red')
        plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: J[c[0]][2] == 1, enumerate(I)))))[:, 0],
                    np.array(list(map(lambda elt: elt[1], filter(lambda c: J[c[0]][2] == 1, enumerate(I)))))[:, 1],
                    color='green')
        showValue()
    elif test == 6:
        W = np.random.uniform(-1, 1, 3)
        predicted_x1, predicted_x2, predicted_labels = linearModelMultiClass(initWMultiClass(L, K, 0,W), initWMultiClass(L, K, 1, W),initWMultiClass(L, K, 2, W))
        plt.scatter(predicted_x1, predicted_x2, c=predicted_labels)
        plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: L[c[0]][0] == 1, enumerate(K)))))[:, 0],
                    np.array(list(map(lambda elt: elt[1], filter(lambda c: L[c[0]][0] == 1, enumerate(K)))))[:, 1],
                    color='blue')
        plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: L[c[0]][1] == 1, enumerate(K)))))[:, 0],
                    np.array(list(map(lambda elt: elt[1], filter(lambda c: L[c[0]][1] == 1, enumerate(K)))))[:, 1],
                    color='red')
        plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: L[c[0]][2] == 1, enumerate(K)))))[:, 0],
                    np.array(list(map(lambda elt: elt[1], filter(lambda c: L[c[0]][2] == 1, enumerate(K)))))[:, 1],
                    color='green')
        showValue()
    elif test == 7:
        x, y = regressionModel(M, N)
        plt.plot(x, y, color='blue')
        plt.scatter(M, N)
        showValue()
    elif test == 8:
        x, y = regressionModel(O, P)
        plt.plot(x, y, color='blue')
        plt.scatter(O, P)
        showValue()
    elif test == 9:
        x, y = regressionModel(Q, R)
        plt.plot(x, y, color='blue')
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(Q[:, 0], Q[:, 1], R)
        showValue()
    elif test == 10:
        x, y = regressionModel(S, T)
        plt.plot(x, y, color='blue')
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(S[:, 0], S[:, 1], T)
        plt.show()
        plt.clf()
    elif test == 11:
        x, y = regressionModel(U, V)
        plt.plot(x, y, color='blue')
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(U[:, 0], U[:, 1], V)
        plt.show()
        plt.clf()
    else:
        print("Error")


executeTest(7)
