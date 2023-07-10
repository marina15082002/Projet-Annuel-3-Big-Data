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

def tab_to_string(tab) -> str:
    """
    Converts an array of rows of floats to a string.
    Format: "0.1 0.2 0.3\n0.4 0.5 0.6\n0.7 0.8 0.9"
    """
    return "\n".join([" ".join([str(x) for x in row]) for row in tab])

def string_to_tab(string) -> list:
    """
    Converts a string to an array of rows of floats.
    Format: "0.1 0.2 0.3\n0.4 0.5 0.6\n0.7 0.8 0.9"
    """
    return [[float(x) for x in row.split(" ")] for row in string.split("\n")]

def test():
    tab = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    serialized = tab_to_string(tab)
    assert serialized == "0.1 0.2 0.3\n0.4 0.5 0.6\n0.7 0.8 0.9"
    deserialized = string_to_tab(serialized)
    assert deserialized == tab

    serialized = "1 2.0\n3.0 4.0"
    deserialized = string_to_tab(serialized)
    assert deserialized == [[1.0, 2.0], [3.0, 4.0]]



points = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
labels = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)

my_lib = ctypes.CDLL(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\SRC\App\target\debug\App.dll")

my_lib.predict_labels.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
my_lib.predict_labels.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
predicted_data = my_lib.predict_labels(points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(points), len(labels))

# extern "C" fn predict_labels(points: *const f32, labels: *const f32, points_len: i32, labels_len: i32) -> *mut [*mut f32; 3]

result = ctypes.cast(predicted_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))

print(result)

