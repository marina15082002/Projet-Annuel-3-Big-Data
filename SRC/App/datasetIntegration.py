import numpy as np
import ctypes
import matplotlib.pyplot as plt
from PIL import Image
import os


# créer une liste vide
dataset = []  # Créer une liste vide pour stocker les sous-tableaux d'images
nameFile = r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset\Dataset\Pommes\Image_"
i = 1

while os.path.exists(nameFile + str(i) + ".jpg"):
    im = Image.open(nameFile + str(i) + ".jpg")
    image = np.array(im) / 255.0 # / 255.0 pour normaliser les valeurs des pixels entre 0 et 1
    tabImage = image.flatten()
    dataset.append(tabImage)
    i += 1

dataset = np.array(dataset)  # Convertir la liste en tableau numpy

print(dataset[0])