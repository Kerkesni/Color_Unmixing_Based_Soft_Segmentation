import numpy as np
import pickle
from cv2.ximgproc import guidedFilter
import cv2
import argparse
import sys

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

nb_layers = len(distributions)

parser = argparse.ArgumentParser(description='Color Model Estimation')
parser.add_argument('image_path', type=str, nargs='?', help='image path')
parser.add_argument('quality', type=int, nargs='?', default=50, help='image quality level')
args = parser.parse_args()

if(len(sys.argv) < 1):
    print("Not Enough Arguments")
    exit(-1)

# Reading Image
print('Reading Image...')
img = cv2.imread(args.image_path)
percent = args.quality
width = int(img.shape[1] * percent / 100)
height = int(img.shape[0] * percent / 100)
dim = (width, height)
img = cv2.resize(img, dim)

# Guided Filter Params
radius = 5
epsilon = 0.00001

for layer_index in range(nb_layers):
    path = f'res/layer{layer_index+1}.png'
    layer = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    guided = guidedFilter(guide = img, src=layer[:, :, -1], radius=radius, eps=epsilon)
    layer[:, :, -1] = guided
    cv2.imwrite(f"res/reg_{layer_index+1}.png", layer)
