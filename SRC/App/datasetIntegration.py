import numpy as np
import ctypes
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from typing import List

bananes = "Bananes"
pommes = "Pommes"
oranges = "Oranges"
autres = "Autres"

nameBananes = "bananes_"
namePommes = "pommes_"
nameOranges = "oranges_"
nameAutres = "autres_"


# chemin du fichier image a renommer

def rename(name, namefile):
    dirnames = os.path.abspath(os.path.dirname(__file__))
    source_dir = os.path.join(dirnames, '..\..\Dataset\Dataset', name)
    output_dir = os.path.join(dirnames, '..\..\Dataset')

    # liste des fichiers dans le dossier
    files = os.listdir(source_dir)
    existing_nums = []

    # liste des fichiers existants
    files_exist = []

    for file in files:
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            num = file.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').split('_')[-1]
            if num.isdigit():
                num = int(num)
                existing_nums.append(num)

    # compter de numero d image
    num = 1

    # pour chaque fichier dans le dossier
    for file in files:
        # verifier si le fichier est bien une image
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            # chemin complet du fichier
            source_path = os.path.join(source_dir, file)

            # saute les numeros deja existants
            while num in existing_nums:
                num += 1

            # nouveau nom du fichier avec numero sequentiel
            new_name = f'{namefile}{num}.jpg'
            # chelmin complet du nouveau fichier
            new_path = os.path.join(output_dir, new_name)
            # renommer le fichier
            os.rename(source_path, new_path)

            with Image.open(new_path) as img:
                img.resize((32, 32))
            # incrementer de compteur
            num += 1


def expected_image(directory):
    expected = []
    for filename in os.listdir(directory):
        if filename.startswith(nameAutres):
            expected.append([0, 0, 0, 1])
        elif filename.startswith(nameOranges):
            expected.append([0, 0, 1, 0])
        elif filename.startswith(nameBananes):
            expected.append([0, 1, 0, 0])
        elif filename.startswith(namePommes):
            expected.append([1, 0, 0, 0])
        else:
            print(f"Le fichier {filename} n'a pas été étiqueté")

    # convertir les étiquettes en one-hot encoding
    expected = np.array(expected, dtype=np.double)

    return expected


def allcolors(directory):
    all_pixels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            path = os.path.join(directory, filename)
            img = Image.open(path)
            img_resized = img.resize((32, 32))
            img_resized.save(path)
            if img.mode != "RGB":
                print(f"Le fichier {filename} n'est pas en RGB")
                img = img.convert("RGB")
            pixel_list = []
            for pixel in img.getdata():
                r, g, b = pixel
                pixel_list.extend([r / 255, g / 255, b / 255])
            all_pixels.append(pixel_list)
            img.close()
    if all_pixels:
        all_pixels = np.array(all_pixels, dtype=np.double)
        return all_pixels.reshape((all_pixels.shape[0], -1))
    else:
        print("Aucun fichier n'a été trouvé")



expected = expected_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset")
pixels = allcolors(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset")


class MLP:
    def __init__(self):
        self.L = 0
        self.d = []
        self.W = []
        self.X = []
        self.deltas = []

def creatMLP(npl, npl_size):
    mlp = MLP()
    mlp.L = npl_size - 1
    mlp.d = [npl[i] for i in range(mlp.L)]
    mlp.W = [[[random.uniform(-1, 1) for k in range(mlp.d[i] + 1)] for j in range(mlp.d[i + 1])] for i in range(mlp.L - 1)]  # Correction
    mlp.W.append([[random.uniform(-1, 1) for k in range(mlp.d[-1] + 1)]])  # Correction
    mlp.X = [[0.0] * (mlp.d[i] + 1) for i in range(mlp.L)]
    mlp.X.append([0.0] * (mlp.d[-1] + 1))
    mlp.deltas = [[0.0] * mlp.d[i + 1] for i in range(mlp.L - 1)]  # Correction
    mlp.deltas.append([0.0] * mlp.d[-1])  # Correction
    return mlp

# Exemple d'utilisation
npl = [3, 4, 1]  # Couches du réseau de neurones
mlp = creatMLP(npl, len(npl))


def initW(images, labels):
    num_classes = labels.shape[1]
    W = np.random.uniform(-1, 1, (num_classes, images.shape[1] + 1))

    for _ in range(10000):
        k = np.random.randint(0, len(labels))
        Yk = labels[k] - 0.5  # Étiquettes ajustées à l'intervalle [-0.5, 0.5]
        Xk = np.concatenate(([1], images[k]))  # Pixels de l'image
        signal = np.dot(W, Xk)
        gXk = np.where(signal >= 0, 1.0, -1.0)
        W = W + 0.01 * np.outer(Yk - gXk, Xk)

    return W

def linearModel(W, images):
    predicted_scores = []

    for image in images:
        scores = np.dot(W, np.concatenate(([1], image)))
        exp_scores = np.exp(scores)
        probabilities = exp_scores / (np.sum(exp_scores) + np.finfo(float).eps) * 100
        predicted_scores.append(probabilities.tolist())

    return predicted_scores

# Exemple d'utilisation


W = initW(pixels, expected)
scores = linearModel(W, pixels)
"""
print("linear model : ")
print(scores[len(scores) - 180])

print("expected : ")
print(expected[len(expected) - 180])

# affichier la première image de pixels
plt.imshow(pixels[len(expected) - 180].reshape((150, 150, 3)))
plt.show()
"""

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


mlp = myMLP([32*32*3, 10, 4])  # Création d'une instance du modèle MLP avec les couches spécifiées

# Entraînement du modèle avec des données d'entraînement
mlp.train(pixels, expected, True, 1000, 0.01)
"""
# Prédiction sur une image
prediction = mlp.predict(pixels[len(pixels) - 180], True)
print(prediction)
print("expected : ")
print(expected[len(expected) - 180])
"""


class RBFModel:
    def __init__(self, num_classes, num_centers, gamma):
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.gamma = gamma
        self.centers = None
        self.weights = None

    def fit(self, X, y):
        # Sélection aléatoire des centres
        random_indices = np.random.choice(X.shape[0], size=self.num_centers, replace=False)
        self.centers = X[random_indices]

        # Calcul des activations pour chaque échantillon
        activations = self.calculate_activations(X)

        # Ajout d'une colonne de biais aux activations
        activations_bias = np.hstack((np.ones((X.shape[0], 1)), activations))

        self.weights = np.zeros((activations_bias.shape[1], y.shape[1]))

        for i in range(activations_bias.shape[1]):
            for j in range(y.shape[1]):
                total = 0.0
                for k in range(activations_bias.shape[0]):
                    total += activations_bias[k, i] * y[k, j]
                self.weights[i, j] = total

        # Mise à l'échelle des poids entre 0 et 100
        max_weight = np.max(self.weights)
        min_weight = np.min(self.weights)
        self.weights = 100 * (self.weights - min_weight) / (max_weight - min_weight)

    def predict(self, X):
        activations = self.calculate_activations(X)
        activations_bias = np.hstack((np.ones((X.shape[0], 1)), activations))
        predictions = activations_bias @ self.weights
        return predictions

    def calculate_activations(self, X):
        num_samples = X.shape[0]
        num_centers = self.centers.shape[0]
        activations = np.zeros((num_samples, num_centers))

        for i in range(num_samples):
            for j in range(num_centers):
                diff = X[i] - self.centers[j]
                activations[i, j] = np.exp(-self.gamma * np.dot(diff, diff))

        return activations

num_classes = 4  # Nombre de classes (Pommes, Bananes, Oranges, Autres)
num_centers = 10  # Nombre de centres RBF
gamma = 0.1  # Paramètre gamma pour les fonctions de base radiales
rbf_model = RBFModel(num_classes, num_centers, gamma)

# Entraînement du modèle avec les données d'entraînement
rbf_model.fit(pixels, expected)

predictions = rbf_model.predict(pixels[0])
percentages = predictions.flatten()
print(percentages)

print("expected : ")
print(expected[0])