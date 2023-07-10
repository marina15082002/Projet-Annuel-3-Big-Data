import numpy as np

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