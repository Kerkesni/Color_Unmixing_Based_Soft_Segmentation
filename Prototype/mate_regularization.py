import numpy as np
import pickle
from guided_filter import guided_filter

# Helper Function that map [0, 255] to [0, 1]
def translate(value, leftMin=0, leftMax=255, rightMin=0, rightMax=1):

    if value > leftMax:
        value = leftMax
    if value < leftMin:
        value = leftMin

    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

# Reading Image
print('Reading Image...')
img = cv2.imread('assets/d.jpg')
percent = 20
width = int(img.shape[1] * percent / 100)
height = int(img.shape[0] * percent / 100)
dim = (height, width)
img = cv2.resize(img, dim)

# Retreiving layer's original shape
print('retreiving shape...')
shape = (img.shape[0], img.shape[1])

# Loading Alpha Layer Values
alpha_layers = pickle.load(open('./data/alpha_min.dat', 'rb'))

# Guided Filter Params
radius = 5
epsilon = 0.0001
######################

# Translating values
for i in range (len(alpha_layers)):
    alpha_layers[i] = translate(alpha_layers[i], 0, 1, 0, 255).astype(int)

# reshaping to original shape
alpha_layers = np.reshape(alpha_layers, shape)

# Applying Guided Filter
new_layers = np.empty_like(alpha_layers)
for layer in alpha_layers:
    filtered_layer = guided_filter(layer, layer, radius, epsilon)
    new_layers = filtered_layer

# TODO Normalize sum of alpha values for each pixel
