from image import *
from mlp import *
from modele_linaire import *
from create_graph import *
from rbf import *

FRUITS = ["pommes_", "bananes_", "oranges_", "autres_"]

bananes = "Bananes"
pommes = "Pommes"
oranges = "Oranges"
autres = "Autres"

# rename images
"""
dirnames = os.path.abspath(os.path.dirname(__file__))
source_dir = os.path.join(dirnames, '..\..\Dataset\Dataset', bananes)
output_dir = os.path.join(dirnames, '..\..\Dataset\Train')
rename(FRUITS[1], source_dir, output_dir)
"""

# Création des tableaux d'images et d'étiquettes
expected = expected_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset", FRUITS)
pixels = preprocess_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset")

test_expected = expected_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset\Train", FRUITS)
test_pixels = preprocess_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset\Train")


# Appel mlp
"""
npl = [3, 4, 1]  # Couches du réseau de neurones
mlp = creatMLP(npl, len(npl))
"""


# Appel modele lineaire
"""
W = initW(pixels, expected)
scores = linearModel(W, pixels)

print("linear model : ")
print(scores[len(scores) - 180])

print("expected : ")
print(expected[len(expected) - 180])

# affichier la première image de pixels
plt.imshow(pixels[len(expected) - 180].reshape((150, 150, 3)))
plt.show()
"""


def calculate_accuracy(predictions, expected_outputs):
    correct_count = 0
    total_count = len(predictions)

    for prediction, expected in zip(predictions, expected_outputs):
        if np.argmax(prediction) == np.argmax(expected):
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

# Entraînement du modèle avec des données d'entraînement



iteration_count = 10
alpha = 0.01

W = initW(pixels, expected)

# TEST POUR FAIRE LE GRAPHIQUE POUR LE MODELE LINEAIRE DE LA MEME FACON QUE LE PROF
# Listes pour stocker les précisions d'apprentissage et de test à chaque itération
train_accuracy_linear_list = []
test_accuracy_linear_list = []

train_error_linear_list = []
test_error_linear_list = []

epochs = range(1, iteration_count + 1)

def calculate_error(predictions, targets):
    errors = np.abs(predictions - targets)
    average_error = np.mean(errors)
    return average_error

def calculate_loss(predictions, targets):
    squared_errors = (predictions - targets) ** 2
    loss = np.mean(squared_errors)
    return loss

def compute_gradients(predictions, targets, inputs, W):
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = predictions - targets
    gradients = np.outer(errors, inputs) / len(targets)
    return gradients


train_loss_linear_list = []
test_loss_linear_list = []
epochs = range(1, iteration_count + 1)
momentum = 0.9
velocity = np.zeros_like(W)
learning_rate = 0.01
"""
for it in range(iteration_count):
    # Entraînement du modèle linéaire avec les échantillons d'entraînement
    train_predictions_linear = linearModel(W, pixels)
    train_loss_linear = calculate_loss(train_predictions_linear, expected)
    train_loss_linear_list.append(train_loss_linear)

    # Calcul des prédictions et de la perte de test
    test_predictions_linear = linearModel(W, test_pixels)
    test_loss_linear = calculate_loss(test_predictions_linear, test_expected)
    test_loss_linear_list.append(test_loss_linear)

    # Sélection aléatoire des indices du mini-lot
    batch_size = 32  # Taille du mini-lot
    indices = np.random.choice(len(expected), size=batch_size, replace=False)
    batch_predictions_linear = [train_predictions_linear[i] for i in indices]
    batch_expected = [expected[i] for i in indices]
    batch_pixels = [pixels[i] for i in indices]

    # Calcul du gradient
    gradients = compute_gradients(batch_predictions_linear, batch_expected, batch_pixels, W)

    gradients = gradients.reshape(W.shape)

    # Mise à jour des poids avec moment
    velocity = momentum * velocity - learning_rate * np.tile(gradients, (W.shape[0], 1))
    W += velocity
"""
# Fonction pour afficher la courbe de perte


# Affichage du graphique de perte d'apprentissage et de test pour le modèle linéaire
#plot_loss_curve(train_loss_linear_list, test_loss_linear_list, epochs)

# TEST POUR FAIRE LE GRAPHIQUE POUR LE MODELE LINEAIRE EN FONCTION DU POURCENTAGE

"""
# Boucle d'entraînement
for it in range(iteration_count):
    # Entraînement du modèle linéaire avec les échantillons d'entraînement
    train_predictions_linear = linearModel(W, pixels)
    train_accuracy_linear = calculate_accuracy(train_predictions_linear, expected)
    train_accuracy_linear_list.append(train_accuracy_linear)

    # Calcul des prédictions et de la précision de test
    test_predictions_linear = linearModel(W, test_pixels)
    test_accuracy_linear = calculate_accuracy(test_predictions_linear, test_expected)
    test_accuracy_linear_list.append(test_accuracy_linear)

    # Mise à jour des poids W en utilisant la fonction initW
    W = initW(pixels, expected)

# Affichage du graphique des réussites d'apprentissage et de test pour le modèle linéaire
plot_learning_curve(train_accuracy_linear_list, test_accuracy_linear_list)

# Affichage du graphique des réussites d'apprentissage et de test pour le modèle linéaire
plot_learning_curve(train_accuracy_linear_list, test_accuracy_linear_list)

"""
# Utilisation des fonctions pour prétraiter l'image et faire la prédiction

image_path = r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset\Train\oranges_49.jpg"  # Remplacez par le chemin de votre image
preprocessed_image = preprocess_image(image_path)
prediction_scores = linearModel(W, [preprocessed_image])
print(prediction_scores)
predicted_class_index = np.argmax(prediction_scores[0])
print(predicted_class_index)
class_labels = ["pomme", "banane", "orange", "autre"]
predicted_class = class_labels[predicted_class_index]

# Afficher la prédiction
print("L'image est prédite comme :", predicted_class)



# TEST POUR FAIRE LE GRAPHIQUE POUR LE PMC EN FONCTION DU POURCENTAGE
train_accuracy = []
test_accuracy = []

iteration_count = 10
alpha = 0.01

mlp = myMLP([32 * 32 * 3, 10, 4])  # Création d'une instance du modèle MLP avec les couches spécifiées

# Boucle d'entraînement
for it in range(iteration_count):
    # Entraînement du PMC avec les échantillons d'entraînement
    mlp.train(pixels, expected, True, 1, alpha)

    # Calcul de la précision d'apprentissage
    train_predictions = []
    for inputs in pixels:
        prediction = mlp.predict(inputs, True)
        train_predictions.append(prediction)
    train_accuracy.append(calculate_accuracy(train_predictions, expected))

    # Calcul de la précision de test
    test_predictions = []
    for test_inputs, test_outputs in zip(test_pixels, test_expected):
        prediction = mlp.predict(test_inputs, True)
        test_predictions.append(prediction)
    test_accuracy.append(calculate_accuracy(test_predictions, test_expected))

# Affichage du graphique des réussites d'apprentissage et de test
plot_learning_curve(train_accuracy, test_accuracy)

# Affichage du graphique des réussites d'apprentissage et de test
plot_learning_curve(train_accuracy, test_accuracy)



# Prédiction sur une image
prediction = mlp.predict(pixels[len(pixels) - 180], True)
print(prediction)
print("expected : ")
print(expected[len(expected) - 180])


# test RBF
num_classes = 4  # Nombre de classes (Pommes, Bananes, Oranges, Autres)
num_centers = 10  # Nombre de centres RBF
gamma = 0.1  # Paramètre gamma pour les fonctions de base radiales
rbf_model = RBFModel(num_classes, num_centers, gamma)

# Entraînement du modèle avec les données d'entraînement
rbf_model.fit(pixels, expected)

predictions = rbf_model.predict(pixels[0])
percentages = predictions.flatten()
"""
print(percentages)

print("expected : ")
print(expected[0])
"""
