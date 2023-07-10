from image import *
from mlp import *
from modele_linaire import *
from create_graph import *
from rbf import *

FRUITS = ["pommes_", "bananes_", "oranges_", "autres_"]

# Création des tableaux d'images et d'étiquettes
expected = expected_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset", FRUITS)
pixels = preprocess_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset")

test_expected = expected_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset\Train", FRUITS)
test_pixels = preprocess_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset\Train")

mlp = myMLP([32 * 32 * 3, 10, 4])  # Création d'une instance du modèle MLP avec les couches spécifiées

# Entraînement du modèle
train_loss = []
test_loss = []
epochs = 50  # Nombre d'époques d'entraînement
alpha = 0.1  # Taux d'apprentissage

def calculate_accuracy(predictions, expected_outputs):
    correct_count = 0
    total_count = len(predictions)

    for prediction, expected in zip(predictions, expected_outputs):
        if np.argmax(prediction) == np.argmax(expected):
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

for epoch in range(epochs):
    mlp.train(pixels, expected, True, epochs, alpha)  # Entraînement du modèle pour une époque

    # Calcul du loss d'apprentissage et de validation
    train_loss.append(mlp._calculate_loss(pixels, expected, True))
    test_loss.append(mlp._calculate_loss(test_pixels, test_expected, True))

    # afficher le pourcentage d'erreur pour chaque epoch pour le train et le test
    """
    train_predictions = []
    for inputs in pixels:
        prediction = mlp.predict(inputs, True)
        train_predictions.append(prediction)
    train_accuracy = calculate_accuracy(train_predictions, expected)
    print("train accuracy : ", train_accuracy)

    test_predictions = []
    for test_inputs, test_outputs in zip(test_pixels, test_expected):
        prediction = mlp.predict(test_inputs, True)
        test_predictions.append(prediction)
    test_accuracy = calculate_accuracy(test_predictions, test_expected)
    print("test accuracy : ", test_accuracy)
    """


# Affichage du graphique du loss
plt.plot(range(epochs), train_loss, label='Training Loss')
plt.plot(range(epochs), test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# prédiction pour une image
prediction = mlp.predict(test_pixels[2], True)
print(prediction)
print(test_expected[2])

prediction = mlp.predict(test_pixels[28], True)
print(prediction)
print(test_expected[28])

prediction = mlp.predict(test_pixels[70], True)
print(prediction)
print(test_expected[70])

prediction = mlp.predict(test_pixels[52], True)
print(prediction)
print(test_expected[52])





