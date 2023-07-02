import numpy as np
import ctypes
import matplotlib.pyplot as plt
from typing import List

A = [
    [1, 1],
    [2, 3],
    [3, 3]
]
B = [
    [1],
    [-1],
    [-1]
]

C = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
D = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

E = [[1, 0], [0, 1], [0, 0], [1, 1]]
F = [[1], [1], [-1], [-1]]

G = np.random.random((500, 2)) * 2.0 - 1.0
H = [[1] if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else [-1] for p in G]

I = np.random.random((500, 2)) * 2.0 - 1.0
J = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
              [0, 0, 0] for p in I])

I = I[[not np.all(arr == [0, 0, 0]) for arr in J]]
J = J[[not np.all(arr == [0, 0, 0]) for arr in J]]

K = np.random.random((1000, 2)) * 2.0 - 1.0
L = [[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
    p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in K]

M = [
    [1],
    [2]
]
N = [
    [2],
    [3]
]

O = [
    [1],
    [2],
    [3]
]
P = [
    [2],
    [3],
    [2.5]
]

Q = [
    [1, 1],
    [2, 2],
    [3, 1]
]
R = [
    [2],
    [3],
    [2.5]
]

S = [
    [1, 1],
    [2, 2],
    [3, 3]
]
T = [
    [1],
    [2],
    [3]
]

U = [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
]
V = [
    [2],
    [1],
    [-2],
    [-1]
]

class myMLP:
    def __init__(self, npl: List[int]):
        self.d = npl
        self.L = len(npl) - 1
        self.W = []

        for l in range(self.L + 1):
            self.W.append([])

            if l == 0:
                continue
            for i in range(npl[l - 1] + 1):
                self.W[l].append([])
                for j in range(npl[l] + 1):
                    self.W[l][i].append(0.0 if j == 0 else np.random.uniform(-1.0, 1.0))

        self.X = []
        for l in range(self.L + 1):
            self.X.append([])
            for j in range(npl[l] + 1):
                self.X[l].append(1.0 if j == 0 else 0.0)

        self.deltas = []
        for l in range(self.L + 1):
            self.deltas.append([])
            for j in range(npl[l] + 1):
                self.deltas[l].append(0.0)


    def _propagate(self, inputs: List[float], is_classification: bool):
        for j in range(self.d[0]):
            self.X[0][j + 1] = inputs[j]

        for l in range(1, self.L + 1):
            for j in range(1, self.d[l] + 1):
                total = 0.0
                for i in range(0, self.d[l - 1] + 1):
                    total += self.W[l][i][j] * self.X[l - 1][i]

                if l < self.L or is_classification:
                    total = np.tanh(total)

                self.X[l][j] = total


    def predict(self, inputs: List[float], is_classification: bool):
        self._propagate(inputs, is_classification)
        return self.X[self.L][1:]


    def train(self, all_samples_inputs: List[List[float]], all_samples_expected_outputs: List[List[float]],
              is_classification: bool, iteration_count: int, alpha: float):
        for it in range(iteration_count):
            k = np.random.randint(0, len(all_samples_inputs))
            inputs_k = all_samples_inputs[k]
            y_k = all_samples_expected_outputs[k]

            self._propagate(inputs_k, is_classification)

            for j in range(1, self.d[self.L] + 1):
                self.deltas[self.L][j] = (self.X[self.L][j] - y_k[j - 1])
                if is_classification:
                    self.deltas[self.L][j] *= (1 - self.X[self.L][j] ** 2)

            for l in reversed(range(1, self.L + 1)):
                for i in range(1, self.d[l - 1] + 1):
                    total = 0.0
                    for j in range(1, self.d[l] + 1):
                        total += self.W[l][i][j] * self.deltas[l][j]
                    self.deltas[l - 1][i] = (1 - self.X[l - 1][i] ** 2) * total

            for l in range(1, self.L + 1):
                for i in range(0, self.d[l - 1] + 1):
                    for j in range(1, self.d[l] + 1):
                        self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j]


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
        mlp = myMLP([2, 1])
        mlp.train(A, B, True, 100000, 0.01)
        for sample_inputs in A:
            print(mlp.predict(sample_inputs, True))
    elif test == 2:
        # Cas de test 2
        mlp = myMLP([2, 1])
        mlp.train(C, D, True, 100000, 0.01)
        for sample_inputs in C:
            print(mlp.predict(sample_inputs, True))
    elif test == 3:
        # Cas de test 3
        mlp = myMLP([2, 2, 1])
        mlp.train(E, F, True, 100000, 0.01)
        for sample_inputs in E:
            print(mlp.predict(sample_inputs, True))
    elif test == 4:
        # Cas de test 4
        mlp = myMLP([2, 4, 1])
        mlp.train(G, H, True, 100000, 0.01)
        for sample_inputs in G:
            print(mlp.predict(sample_inputs, True))
    elif test == 5:
        # Cas de test 5
        mlp = myMLP([2, 3])
        mlp.train(I, J, True, 100000, 0.01)
        for sample_inputs in I:
            print(mlp.predict(sample_inputs, True))
    elif test == 6:
        mlp = myMLP([2, 1, 1, 3])
        mlp.train(K, L, True, 100000, 0.01)
        for sample_inputs in K:
            print(mlp.predict(sample_inputs, True))
    elif test == 7:
        mlp = myMLP([1, 1])
        mlp.train(M, N, True, 100000, 0.01)
        for sample_inputs in M:
            print(mlp.predict(sample_inputs, True))
    elif test == 8:
        mlp = myMLP([1, 1, 1])
        mlp.train(O, P, True, 100000, 0.01)
        for sample_inputs in O:
            print(mlp.predict(sample_inputs, True))
    elif test == 9:
        mlp = myMLP([2, 1])
        mlp.train(Q, R, True, 100000, 0.01)
        for sample_inputs in Q:
            print(mlp.predict(sample_inputs, True))
    elif test == 10:
        mlp = myMLP([2, 1])
        mlp.train(S, T, True, 100000, 0.01)
        for sample_inputs in S:
            print(mlp.predict(sample_inputs, True))
    elif test == 11:
        mlp = myMLP([2, 2, 1])
        mlp.train(U, V, True, 100000, 0.01)
        for sample_inputs in U:
            print(mlp.predict(sample_inputs, True))
    else:
        print("Error")


executeTest(3)
