import numpy as np
import ctypes
import matplotlib.pyplot as plt

points = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
labels = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)

my_lib = ctypes.CDLL(r"C:\Users\marin\Documents\3IABD\PA\Projet-Annuel-3-Big-Data\SRC\App\target\debug\App.dll")

my_lib.predict_labels.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
my_lib.predict_labels.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
predicted_data = my_lib.predict_labels(points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(points), len(labels))

# extern "C" fn predict_labels(points: *const f32, labels: *const f32, points_len: i32, labels_len: i32) -> *mut [*mut f32; 3]

result = ctypes.cast(predicted_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))

predicted_x1_ptr = ctypes.cast(result[0], ctypes.POINTER(ctypes.c_float))
predicted_x2_ptr = ctypes.cast(result[1], ctypes.POINTER(ctypes.c_float))
predicted_labels_ptr = ctypes.cast(result[2], ctypes.POINTER(ctypes.c_float))

print(predicted_x1_ptr)
print(predicted_labels_ptr)

# Conversion des pointeurs en tableaux NumPy
predicted_x1 = np.ctypeslib.as_array(predicted_x1_ptr, shape=(90000,))
predicted_x2 = np.ctypeslib.as_array(predicted_x2_ptr, shape=(90000,))
predicted_labels = np.ctypeslib.as_array(predicted_labels_ptr, shape=(90000,))



# Affichage des valeurs prédites
"""for x1, x2, label in zip(predicted_x1.flat, predicted_x2.flat, predicted_labels.flat):
    print(f"x1: {x1}, x2: {x2}, label: {label}")
"""
# Affichage des points prédits

#predicted_labels_len = len(predicted_labels)
"""for i in range(predicted_labels_len):
    if predicted_labels[i] == 1:
        predicted_labels[i] = 'pink'
    else:
        predicted_labels[i] = 'lightskyblue'

plt.scatter(predicted_x1, predicted_x2, c=predicted_labels)"""


"""
points.ctypes.free()
labels.ctypes.free()
np.ctypeslib.as_array(predicted_data, shape=(2 * len(points) + 1)).base.free()
"""

"""
plt.scatter(points[1:3,0], points[1:3,1], color='blue')
plt.scatter(points[0, 0], points[0, 1], color='red')
"""
couleurs = []
for i in range(len(predicted_labels)):
    if predicted_labels[i] != 0.0000000e+00:
        couleurs.append('lightskyblue')
    else:
        couleurs.append('pink')

plt.scatter(predicted_x1, predicted_x2, c=couleurs)
plt.scatter(points[0:2, 0], points[0:2, 1], color='blue')
plt.scatter(points[2:4, 0], points[2:4, 1], color='red')

plt.show()
plt.clf()
