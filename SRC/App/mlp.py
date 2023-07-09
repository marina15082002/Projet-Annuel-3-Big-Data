import random
from typing import List
import numpy as np

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

    def _calculate_loss(self, inputs, expected_outputs, is_classification):
        loss = 0.0
        for i in range(len(inputs)):
            predicted_outputs = self.predict(inputs[i], is_classification)
            if is_classification:
                loss += sum((np.array(predicted_outputs) - np.array(expected_outputs[i])) ** 2)
            else:
                loss += sum(np.abs(np.array(predicted_outputs) - np.array(expected_outputs[i])))
        return loss / len(inputs)

    def _compute_loss(self, inputs, expected_outputs, validation_data=None, is_classification=True):
        train_loss = self._calculate_loss(inputs, expected_outputs, is_classification)
        print(f"Training Loss: {train_loss}")

        if validation_data:
            val_loss = self._calculate_loss(validation_data[0], validation_data[1], is_classification)
            print(f"Validation Loss: {val_loss}")