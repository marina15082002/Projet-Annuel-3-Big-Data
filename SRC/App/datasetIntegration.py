import numpy as np
import ctypes
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

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
            img = Image.open(os.path.join(directory, filename))
            if img.mode != "RGB":
                print(f"Le fichier {filename} n'est pas en RGB")
                img = img.convert("RGB")
            pixel_list = []
            for pixel in img.getdata():
                r, g, b = pixel
                pixel_list.extend([r / 255, g / 255, b / 255])
            all_pixels.append(pixel_list)
    if all_pixels:
        all_pixels = np.array(all_pixels, dtype=np.double)
        return all_pixels.reshape((all_pixels.shape[0], -1))
    else:
        print("Aucun fichier n'a été trouvé")



expected = expected_image(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset")
pixels = allcolors(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset")

print(len(pixels))

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
print(mlp.L, mlp.d, mlp.W, mlp.X, mlp.deltas)



# créer une liste vide

# dataset = []  # Créer une liste vide pour stocker les sous-tableaux d'images
# nameFile = r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\Dataset\Dataset\Pommes\image_"
# i = 1
#
# while os.path.exists(nameFile + str(i) + ".jpg"):
#     im = Image.open(nameFile + str(i) + ".jpg")
#     image = np.array(im) / 255.0  # / 255.0 pour normaliser les valeurs des pixels entre 0 et 1
#     tabImage = image.flatten()
#     dataset.append(tabImage)
#     i += 1
#
# dataset = np.array(dataset)  # Convertir la liste en tableau numpy
