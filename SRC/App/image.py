import numpy as np
from PIL import Image
import os

def rename(namefile, source, output):
    files = [f for f in os.listdir(source) if f.endswith(('.jpg', '.png', '.jpeg'))]
    existing_nums = []

    for file in files:
        num = file.split('_')[-1].split('.')[0]
        if num.isdigit():
            existing_nums.append(int(num))

    num = 1

    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            source_path = os.path.join(source, file)

            while num in existing_nums:
                num += 1

            new_name = f'{namefile}{num}.jpg'
            new_path = os.path.join(output, new_name)
            os.rename(source_path, new_path)

            with Image.open(new_path) as img:
                if img.size != (32, 32):
                    img_resized = img.resize((32, 32))
                    img_resized.save(new_path)

            num += 1

def expected_image(directory, fruits):
    expected = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            if filename.startswith(fruits[0]):
                expected.append([1, 0, 0, 0])
            elif filename.startswith(fruits[1]):
                expected.append([0, 1, 0, 0])
            elif filename.startswith(fruits[2]):
                expected.append([0, 0, 1, 0])
            elif filename.startswith(fruits[3]):
                expected.append([0, 0, 0, 1])
            else:
                print(f"L\'image {filename} n'a pas été étiqueté")

    # convertir les étiquettes en one-hot encoding
    expected = np.array(expected, dtype=np.double)

    return expected

def preprocess_image(directory):
    all_pixels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            path = os.path.join(directory, filename)
            img = Image.open(path)

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
