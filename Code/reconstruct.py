import cv2
import numpy as np

# Helper Function that map [0, 255] to [0, 1]
def translate(value, reversed = False):
    if not reversed:
        return value /255
    return np.rint(value*255)

# Translate vector to range
def translate_vec(v, reversed = False):
    tmp = np.empty_like(v, dtype=float)
    for index in range(len(v)):
        tmp[index] = translate(v[index], reversed)
    return tmp

data = open('./data/dist.dat', 'rb')
distributions = pickle.load(data)

root_folder = "res/ex8/"
nb_layers = len(distributions)

original_image = cv2.imread('../assets/m.jpg')
percent = 50
width = int(original_image.shape[1] * percent / 100)
height = int(original_image.shape[0] * percent / 100)
dim = (width, height)
original_image = cv2.resize(original_image, dim)

reconstructed_image = np.zeros_like(original_image, dtype="float64")

for layer_index in range(nb_layers):
    path = f'{root_folder}layer{layer_index+1}_ref.png'
    layer = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    for j in range(layer.shape[0]):
        for i in range(layer.shape[1]):
            color = translate(layer[j, i, 3]) * layer[j, i, 0:3]
            reconstructed_image[j,i] += color

cv2.imwrite(f"{root_folder}reconstructed.jpg", reconstructed_image)
