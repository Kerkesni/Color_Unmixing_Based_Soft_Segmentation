import math
import sys
import multiprocessing
import numpy as np
from scipy.spatial.distance import mahalanobis, euclidean
from optimize import fmin_cg
import cv2
import pickle
import argparse

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

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

# Calculates Energy for a pixel
def EnergyFunction(x, distributions):
    a = x[0:len(distributions)]
    a = translate_vec(a)
    u = x[len(distributions):].reshape(-1, 3)
    F = 0
    for layer_index in range(len(a)):
        # Calculating mahalanobis distance
        Di = mahalanobis(u[layer_index], translate_vec(distributions[layer_index][0]), distributions[layer_index][1])
        F += a[layer_index]*Di

    return F

# Alpha Deviation Constraint
def Ga(x, nb_dist, reg_alphas):
    G = 0
    alpha = x[0:nb_dist]
    for i in range(len(alpha)):
        G += math.pow(alpha[i] - reg_alphas[i], 2)
    return G

# Cplor Deviation Constraint
def Gu(x, img_point, nb_dist):
    G = 0
    a = x[0:nb_dist]
    u = x[nb_dist:].reshape(-1, 3)
    for index in range(len(a)):
            G += a[index]*u[index]
    G = np.square(G - img_point)
    return G

# Global Deviation Constraint
def Gg(x, img_point, nb_dist, reg_alphas):
    img_point = translate_vec(img_point)
    return np.append(Gu(x, img_point, nb_dist), Ga(x, nb_dist, reg_alphas))

# Returns initial alpha and color value for pixel
def get_x(coords, color, alpha, distributions):
    a = np.array([])
    u = np.array([])
    for index in range(len(distributions)):
        a = np.append(a, alpha[index, coords[0], coords[1]])
        u = np.append(u, color[index, coords[0], coords[1]])

    x = np.append(a, u)
    return x

# Original Method of Multipliers for each pixel
def pixel_minimize_function(img_point, x, distributions, reg_alphas):
    rho = 0.1
    lmbda = np.array([0.1, 0.1, 0.1, 0.1])
    beta = 10
    gamma = 0.25
    eps = 0.1

    # Function Minimized Using Conjugated Gradient
    def funcToMinimize(x):
        F = EnergyFunction(x, distributions)
        Gk = Gg(x, img_point, len(distributions), reg_alphas)
        res = np.linalg.norm(F + lmbda.T * Gk + 1/2 * rho * np.square(np.linalg.norm(Gk)))
        return res

    Gk = Gg(x, img_point, len(distributions))
    while(True):
        # minimization
        output = fmin_cg(funcToMinimize, x, disp=False, full_output=True)
        Xkpls1 = output[0]

        if np.linalg.norm(Xkpls1-x) <= eps:
            return Xkpls1
        else:
            x = Xkpls1
            Ggkpls1 = Gg(Xkpls1, img_point, len(distributions), reg_alphas)
            lmbda = lmbda + rho * Ggkpls1
            rho = beta*rho if np.linalg.norm(Ggkpls1) > gamma*np.linalg.norm(Gk) else rho
            Gk = Ggkpls1

# Original Method of Multipliers per chunck of image
def chunk_minimize_function(img, start_index, shared_color_layers, shared_alpha_layers, original_dim_color, original_dim_alpha, distributions, alpha_layers_reg):

    # reshaping the arrays from 1d to 3d
    color_layers = np.frombuffer(shared_color_layers.get_obj())
    color_layers = color_layers.reshape(original_dim_color)

    alpha_layers = np.frombuffer(shared_alpha_layers.get_obj())
    alpha_layers = alpha_layers.reshape(original_dim_alpha)

    idx = 0

    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            x = get_x((j,i), color_layers, alpha_layers, distributions)
            reg_alphas = alpha_layers_reg[:, j, i]
            values = pixel_minimize_function(img[j,i], x, distributions, reg_alphas)
            alphas = values[0:len(distributions)]
            colors = values[len(distributions):].reshape((-1,3))
            for index in range(len(alpha_layers)):
                alpha_layers[index][start_index+j,i] = alphas[index]
                color_layers[index][start_index+j,i] = colors[index]

            idx += 1
            printProgressBar(idx, img.shape[0]*img.shape[1])


if __name__ == "__main__":

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
    cv2.imwrite('./res/original.jpg', img)

    print('Retreiving Distributions...')
    data = open('./data/dist.dat', 'rb')
    distributions = pickle.load(data)

    original_dim_color = (len(distributions), img.shape[0], img.shape[1], 3)
    original_dim_alpha = (len(distributions), img.shape[0], img.shape[1], 1)

    # Retreiving data
    print("retreiving data")
    color_layers = pickle.load(open('./data/color_min.dat', "rb"))
    alpha_layers = pickle.load(open('./data/alpha_min.dat', "rb"))
    alpha_layers_reg = pickle.load(open('./data/alpha_min_reg.dat', "rb"))
    alpha_layers_reg = alpha_layers_reg.reshape(original_dim_alpha)

    shared_color_layers = multiprocessing.Array('d', color_layers)
    shared_alpha_layers = multiprocessing.Array('d', alpha_layers)

    # Color & Alpha Minimization
    print('Minimizing Energy Function For Each Pixel...')

    nb_jobs = 4
    step = math.ceil(img.shape[0]/nb_jobs)
    jobs = []
    increment = 0
    for job in range(nb_jobs):
        start = increment
        finish = start+step if (start+step) <= img.shape[0] else img.shape[0]
        p = multiprocessing.Process(target=chunk_minimize_function, args=(img[start:finish,:], start, shared_color_layers, shared_alpha_layers, original_dim_color, original_dim_alpha, distributions, alpha_layers_reg))
        jobs.append(p)
        p.start()
        increment += step

    for job in jobs:
        job.join()


    print('Reshaping minimization results...')
    color_layers = np.copy(shared_color_layers.get_obj())
    alpha_layers = np.copy(shared_alpha_layers.get_obj())

    # Storing values
    print('Storing results...')
    out_c = open('./data/color_min_ref.dat', "wb")
    out_a = open('./data/alpha_min_ref.dat', "wb")
    pickle.dump(color_layers, out_c)
    pickle.dump(alpha_layers, out_a)

    # color_layers = pickle.load(open('./data/color_min.dat', "rb"))
    # alpha_layers = pickle.load(open('./data/alpha_min.dat', "rb"))

    # Translating values from [0, 1] to [0, 255]
    for i in range (len(color_layers)):
        color_layers[i] = translate(color_layers[i], True)

    for i in range (len(alpha_layers)):
        alpha_layers[i] = translate(alpha_layers[i], True)

    color_layers = color_layers.reshape(original_dim_color) 
    alpha_layers = alpha_layers.reshape(original_dim_alpha)

    for index in range(len(distributions)):
        layer = np.empty((img.shape[0], img.shape[1], 4))
        for j in range(layer.shape[0]):
            for i in range(layer.shape[1]):
                layer[j,i] = np.array([color_layers[index][j,i][0], color_layers[index][j,i][1], color_layers[index][j,i][2], alpha_layers[index][j,i]])
        
        cv2.imwrite(f'./res/layer{index+1}_ref.png', layer)