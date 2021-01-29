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

    top = 0
    bottom = 0
    for layer in range(len(a)):
        top += a[layer]
        bottom += math.pow(a[layer], 2)

    if top == 0 or bottom == 0:
        sparcity = 0
    else:
        sparcity = 10*(top/bottom - 1)

    F += sparcity

    return F

# Alpha Deviation Constraint
def Ga(x, nb_dist):
    G = 0
    alpha = x[0:nb_dist]
    for a in alpha:
        G += a
    return math.pow(G-1, 2)

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
def Gg(x, img_point, nb_dist):
    img_point = translate_vec(img_point)
    return np.append(Gu(x, img_point, nb_dist), Ga(x, nb_dist))

# Returns initial alpha and color value for pixel
def get_intitial_x(img_pixel_color, distributions):
    dist = []
    alpha = np.array([])
    color = np.array([])
    
    for index in range(len(distributions)):
        Di = mahalanobis(img_pixel_color, distributions[index][0], distributions[index][1])
        Di += euclidean(img_pixel_color, distributions[index][0])
        dist.append(Di)

    best_fiting_distribution_index = np.argmin(dist)

    # Setting alpha and color values for layers
    for index in range(len(distributions)):
        if index == best_fiting_distribution_index:
            alpha = np.append(alpha, 1)
            color = np.append(color, translate_vec(img_pixel_color))
        else:
            alpha = np.append(alpha, 0)
            color = np.append(color, translate_vec(distributions[index][0]))

    color = color.flatten()
    x = np.append(alpha, color)
    return x

# Original Method of Multipliers for each pixel
def pixel_minimize_function(img_point, x, distributions):
    rho = 0.1
    lmbda = np.array([0.1, 0.1, 0.1, 0.1])
    beta = 10
    gamma = 0.25
    eps = 0.01

    # Function Minimized Using Conjugated Gradient
    def funcToMinimize(x):
        F = EnergyFunction(x, distributions)
        Gk = Gg(x, img_point, len(distributions))
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
            Ggkpls1 = Gg(Xkpls1, img_point, len(distributions))
            lmbda = lmbda + rho * Ggkpls1
            rho = beta*rho if np.linalg.norm(Ggkpls1) > gamma*np.linalg.norm(Gk) else rho
            Gk = Ggkpls1

# Original Method of Multipliers per chunck of image
def chunk_minimize_function(img, start_index, shared_color_layers, shared_alpha_layers, original_dim_color, original_dim_alpha, distributions):

    # reshaping the arrays from 1d to 3d
    color_layers = np.frombuffer(shared_color_layers.get_obj())
    color_layers = color_layers.reshape(original_dim_color)

    alpha_layers = np.frombuffer(shared_alpha_layers.get_obj())
    alpha_layers = alpha_layers.reshape(original_dim_alpha)

    idx = 0

    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            x = get_intitial_x(img[j,i], distributions)
            values = pixel_minimize_function(img[j,i], x, distributions)
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

    ######################## Testing ##############################
    # point = (10, 40)
    # x = get_intitial_x(img[point], distributions)
    # x_min = pixel_minimize_function(img[point], x, distributions)

    # print(translate_vec(x, True))
    # print(translate_vec(x_min, True))

    # Display pixel color
    # img_c = np.full((100, 100, 3), translate_vec(img[point]))
    # cv2.imwrite('col.jpg', img_c)
    
    # # Print distribution colors
    # for idx, (mean, cov) in enumerate(distributions):
    #     img_c = np.full((100, 100, 3), translate_vec(mean))
    #     cv2.imwrite(f'{idx}.jpg', img_c)

    ###############################################################

    # Display class of each pixel
    print('Assigning A Class To Each Pixel...')
    img_class = np.empty(img.shape)
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            x = get_intitial_x(img[j,i], distributions)[0:len(distributions)]
            idx = np.argmax(x)
            img_class[j,i] = distributions[idx][0]
    cv2.imwrite('./res/classes.jpg', img_class)

    color_layers = np.empty((len(distributions), img.shape[0], img.shape[1], 3))
    alpha_layers = np.empty((len(distributions), img.shape[0], img.shape[1], 1))

    original_dim_color = color_layers.shape
    original_dim_alpha = alpha_layers.shape

    color_layers = color_layers.flatten()
    alpha_layers = alpha_layers.flatten()

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
        p = multiprocessing.Process(target=chunk_minimize_function, args=(img[start:finish,:], start, shared_color_layers, shared_alpha_layers, original_dim_color, original_dim_alpha, distributions))
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
    out_c = open('./data/color_min.dat', "wb")
    out_a = open('./data/alpha_min.dat', "wb")
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
        
        cv2.imwrite(f'./res/layer{index+1}.png', layer)