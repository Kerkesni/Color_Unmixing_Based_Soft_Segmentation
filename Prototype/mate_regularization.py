import numpy as np
import pickle
from guided_filter import guided_filter
import cv2

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

nb_layers = 5

# Reading Image
print('Reading Image...')
img = cv2.imread('../assets/g.jpg')
percent = 50
width = int(img.shape[1] * percent / 100)
height = int(img.shape[0] * percent / 100)
dim = (width, height)
img = cv2.resize(img, dim)

# Guided Filter Params
radius = 5
epsilon = 0.005

for layer_index in range(nb_layers):
    path = f'res/layer{layer_index+1}.png'
    layer = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    guided = cv2.GaussianBlur(layer, (3,3), 0)
    cv2.imwrite(f"res/reg_{layer_index}.jpg", guided)
