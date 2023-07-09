import numpy as np
import ctypes
import matplotlib.pyplot as plt

def tableau_en_string(tableau):
    tableau_string = ""
    for ligne in tableau:
        ligne_string = ",".join(map(str, ligne))
        tableau_string += ligne_string + "/"
    tableau_string = tableau_string[:-1]  # Supprimer le dernier "/"
    return tableau_string


points = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
labels = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)

my_lib = ctypes.CDLL(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\SRC\App\target\debug\App.dll")

my_lib.predict_labels.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
my_lib.predict_labels.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
predicted_data = my_lib.predict_labels(points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(points), len(labels))

# extern "C" fn predict_labels(points: *const f32, labels: *const f32, points_len: i32, labels_len: i32) -> *mut [*mut f32; 3]

result = ctypes.cast(predicted_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))

print(result)

