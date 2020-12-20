# Soft Color Segmentation Prorotype

# The Algorithme is composed of three stages
# 1. Color Unmixing
# 2. Matte Regularisation
# 3. Color Refinement

import cv2
from cv2.ximgproc import guidedFilter
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import sys
import threading
import multiprocessing
import concurrent.futures

# Global Variables
seeds = []
alpha_layers = []
color_layers = []
distributions = []
representation_scores = []
representation_threshold = 25

# Helper Function that map [0, 255] to [0, 1]
def translate(value, leftMin=0, leftMax=255, rightMin=0, rightMax=1):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

# Returns Per Pixel Vote Results And Winner Bin
def voteOnBin(img, gradient, representation_score, representation_threshold):
    # Voting on the bins
    votes_per_bin = {}
    votes = np.zeros((img.shape[0], img.shape[1], 1))
    nb_votes = 0
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if representation_score[j, i] < representation_threshold:
                continue
            binn = pow(math.e, -gradient[j,i]) * (1 - pow(math.e, -representation_score[j,i]))
            if isinstance(binn, np.ndarray):
                binn = binn[0]
            votes[j,i] = binn
            nb_votes += 1
            if binn in votes_per_bin:
                votes_per_bin[binn] += 1
            else:
                votes_per_bin[binn] = 1

    # Getting the most popular bin
    if nb_votes == 0:
        return -1, [], 0
    votes_per_bin = dict(sorted(votes_per_bin.items(), key=lambda item:item[1], reverse=True))
    winner_bin = list(votes_per_bin.items())[0][0]
    return (winner_bin, votes, nb_votes)

# Returns Pixels In Per Bin
def getPopularBinPixelCoords(votes, bin):
    # We Get All Pixels That Belong To The Bin
    coords = []
    for j in range(votes.shape[0]):
        for i in range(votes.shape[1]):
            if votes[j,i] == bin:
                coords.append((j, i))
    return coords

# Returns Seed Pixel from a bin
def getSeedPixel(img, coords, gradient, size=10):
    # Getting The Next Seed Pixel
    scores = []
    for pixel in coords:
        # 20x20 Kernel Limits
        lower_h = pixel[0]+size if pixel[1]+size < img.shape[1] else img.shape[1]-1
        higher_h = pixel[0]-size if pixel[1]-size >= 0 else 0

        lower_l = pixel[1]-size if pixel[1]-size > 0 else 0
        higher_l = pixel[1]+size if pixel[1]+size < img.shape[1] else img.shape[1]-1

        S = 0
        for j in range(lower_h, higher_h):
            for i in range(lower_l, higher_l):
                if votes[j,i] == winner_bin:
                    S += 1
        score = S*pow(math.e, -gradient[pixel])
    scores.append(score)

    # Getting The Pixel Coordinates With The Higher Score
    seed_coords = coords[np.argmax(scores)]
    return seed_coords

# Returns distribution parameters
def estimateDistribution(img, seed_coords, size=5):
    # Distribution Estimation
    lower_h = seed_coords[0]+size if seed_coords[0]+size < img.shape[1] else img.shape[1]-1
    higher_h = seed_coords[0]-size if seed_coords[0]-size >= 0 else 0

    lower_l = seed_coords[1]-size if seed_coords[1]-size > 0 else 0
    higher_l = seed_coords[1]+size if seed_coords[1]+size < img.shape[1] else img.shape[1]-1

    patch = img[higher_h:lower_h, lower_l:higher_l]
    #dist_weights = guidedFilter(patch, patch, 20, 0.09)

    mean_vector = np.mean(patch.reshape(-1, 3), axis=0)
    covariance_matrix = np.cov(patch.reshape(-1, 3), rowvar=False, bias=True)

    distribution = (mean_vector, covariance_matrix)
    return distribution

# Returns mahalanobis distance of image and distribution
def getMahalanobis(img, distribution, representation_score, representation_threshold=representation_threshold):
    Di = np.full((img.shape[0], img.shape[1]), 255)
    # Foreach Pixel
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if(representation_score[j,i] < representation_threshold):
                continue
            # Inverting Covariance Matrix
            if np.linalg.cond(distribution[1]) < 1/sys.float_info.epsilon:
                inv_cov = np.linalg.inv(distribution[1])
            else:
                inv_cov = distribution[1]
            # Calculating mahalanobis distance
            Di[j,i] = np.minimum(representation_score[j,i], mahalanobis(img[j,i], distribution[0], inv_cov))
    return Di.astype(int)

# Transforms 2d coords into 1d index
def helper_get_index(i, j, size_2d):
    return size_2d[1]*j+i

# Estimating representation score
def estimateRepresentation(img, distributions, representation_score, start, finish, new_representation_score, representation_threshold=representation_threshold):

    start_1d = helper_get_index(0, start, img.shape)

    # Getting the minimum mahalanobis distance
    minDi = getMahalanobis(img, distributions[-1], representation_score)
    minDi_flat = minDi.reshape((-1))

    if len(distributions) < 2 :
        new_representation_score[start_1d:start_1d+len(minDi_flat)] = minDi_flat
        return

    F = np.copy(representation_score)
    for dist1 in range(len(distributions)):
        for dist2 in range(len(distributions)):   
            if dist1 == dist2 :
                continue

            n =  distributions[dist1][0] - distributions[dist2][0]
            for j in range(img.shape[0]):
                for i in range(img.shape[1]):

                    if(representation_score[j,i] < representation_threshold):
                        F[j,i] = representation_score[j,i]
                        continue

                    u1 = img[j,i] - (((img[j,i] - distributions[dist1][0])*n)/n*n)*n
                    u2 = img[j,i] - (((img[j,i] - distributions[dist2][0])*n)/n*n)*n

                    alpha1 = np.linalg.norm(img[j,i] - u1)/np.linalg.norm(u1-u2)
                    alpha2 = 1 - alpha1

                    if np.linalg.cond(distributions[dist1][1]) < 1/sys.float_info.epsilon:
                                inv_cov = np.linalg.inv(distributions[dist1][1])
                    else:
                        inv_cov = distributions[dist1][1]

                    D1 = mahalanobis(u1, distributions[dist1][0], inv_cov)

                    if np.linalg.cond(distributions[dist2][1]) < 1/sys.float_info.epsilon:
                                inv_cov = np.linalg.inv(distributions[dist2][1])
                    else:
                        inv_cov = distributions[dist2][1]

                    D2 = mahalanobis(u2, distributions[dist2][0], inv_cov)
                    tmp_F = alpha1*D1 + alpha2*D2
                    F[j,i] = min(tmp_F, F[j,i], minDi[j,i])
    #return F.astype(int)
    new_representation_score[start_1d:start_1d+len(minDi_flat)] = F.reshape((-1)).astype(int)
    return

# Calculates Energy for a pixel
def EnergyFunction(x, distributions=distributions):
    a = x[0:len(distributions)]
    u = x[len(distributions):].reshape(-1, 3)
    F = 0
    for layer_index in range(len(a)):
        # Inverting Covariance Matrix
        if np.linalg.cond(distributions[layer_index][1]) < 1/sys.float_info.epsilon:
            inv_cov = np.linalg.inv(distributions[layer_index][1])
        else:
            inv_cov = distributions[layer_index][1]
        # Calculating mahalanobis distance
        Di = mahalanobis(u[layer_index], distributions[layer_index][0], inv_cov)
        top = 0
        bottom = 0
        for layer in range(len(a)):
            top += a[layer]
            bottom += math.pow(a[layer], 2)
        sparcity = 10*(top/bottom -1)
        F += a*Di + sparcity
    return F

# Alpha Deviation Constraint
def Ga(x, distributions=distributions):
    G = 0
    alpha = x[0:len(distributions)]
    for a in alpha:
        G += a - 1
    return math.pow(G, 2)

# Cplor Deviation Constraint
def Gu(x, point, img, distributions=distributions):
    G = 0
    a = x[0:len(distributions)]
    u = x[len(distributions):].reshape(-1, 3)
    for index in range(len(a)):
            G += a[index]*u[index]
    G = np.square(G - img[point])
    return G

# Global Deviation Constraint
def Gg(x, point, img):
    return np.append(Gu(x, point, img).T, Ga(x)).T

# Original Method of Multipliers for each pixel
def pixel_minimize_function(img, point, alpha_layers, color_layers):
    a = np.array([])
    u = np.array([])

    for index in range(len(alpha_layers)):
        a = np.append(a, alpha_layers[index][point])
        u = np.append(u, color_layers[index][point].T)

    x = np.append(a, u.T)

    k = 0
    rho = 0.1
    lmbda = np.array([0.1, 0.1, 0.1, 0.1])
    beta = 10
    epsilon = 0.25
    sigma = 5

    # Function Minimized Using Conjugated Gradient (First Line of Algorithm)
    def funcToMinimize(x):
        F = EnergyFunction(x)
        Gk = Gg(x, point, img)
        return min(F + lmbda.T * Gk + 1/2 * rho * np.square(np.linalg.norm(Gk)))

    Gk = Gg(x, point, img)
    while(True):
        minimized_f = minimize(funcToMinimize, x, method='CG')
        Xkpls1 = minimized_f.x
        Xkpls1 = np.array([translate(value) for value in Xkpls1])
        Ggkpls1 = Gg(Xkpls1, point, img)
        lmbdakpls1 = lmbda + rho * Ggkpls1
        rhokpls1 = beta*rho if np.linalg.norm(Ggkpls1) > epsilon*np.linalg.norm(Gk) else rho
        if np.linalg.norm(Xkpls1-x) > sigma:
            k += k+1
            x = Xkpls1
            Gk = Ggkpls1
            lmbda = lmbdakpls1
            rho = rhokpls1
            continue
        else:
            return Xkpls1.astype(int), int(minimized_f.fun)

# Original Method of Multipliers per chunck of image
def chunk_minimize_function(img, alpha_layers, color_layers):
    new_alpha_layers = np.empty((len(alpha_layers), img.shape[0], img.shape[1], 1))
    new_color_layers = np.empty((len(color_layers), img.shape[0], img.shape[1], img.shape[2]))
    new_representation_scores = np.empty((img.shape[0], img.shape[1]))
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            values = pixel_minimize_function(img, (j,i), alpha_layers, color_layers)
            alphas = values[0][0:len(distributions)]
            colors = values[0][len(distributions):].reshape((-1,3))
            for index in range(len(alpha_layers)):
                new_alpha_layers[index][j,i] = alphas[index]
                new_color_layers[index][j,i] = colors[index]
            new_representation_scores[j,i] = int(values[1])
    return new_alpha_layers, new_color_layers, new_representation_scores


if __name__ == '__main__':

    # Reading Image
    print('Reading Image...')
    img = cv2.imread('assets/b.jpg')
    # width = int(img.shape[1] * 50 / 100)
    # height = int(img.shape[0] * 50 / 100)
    # dim = (height, width)
    # img = cv2.resize(img, dim)

    cv2.imwrite('original.jpg', img)

    # Calculating image gradient
    print('Calculating Gradient...')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray,cv2.CV_64F)

    # Initialization
    print('Initializing Layers...')
    alpha_layers.append(np.ones((img.shape[0], img.shape[1], 1)))
    color_layers.append(np.copy(img))
    representation_scores.append(np.full((img.shape[0], img.shape[1], 1), 255))

    first = True
    iteration = 1
    while(True):
        print(f'######## Iteration {iteration} ########')

        # Getting Votes + Most Popular Bin
        print('Voting On Bin...')
        winner_bin, votes, nb_votes = voteOnBin(img, laplacian, representation_scores[-1], representation_threshold)

        # Stoping When all pixels are well represented
        if(nb_votes == 0):
            break

        # Getting Coordinates Of Pixels In The Winner Bin
        print('Retreiving Pixel Coordinates...')
        coords = getPopularBinPixelCoords(votes, winner_bin)

        # Getting The Next Seed Pixel
        print('Finding Seed Pixel...')
        seed_coords = getSeedPixel(img, coords, laplacian)
        seeds.append(seed_coords)

        # Getting Distribution Parameters
        print('Estimating Distribution...')
        distribution = estimateDistribution(img, seed_coords)
        distributions.append(distribution)

        # Setting Ui & Ai
        if not first:
            alpha_layers.append(np.zeros((img.shape[0], img.shape[1], 1)))
            color_layers.append(np.full((img.shape[0], img.shape[1], 3), distribution[0]))


        # Estimate Representation score
        print("Estimating representation scores...")
        # Multiprocessing
        representation_score = np.empty((img.shape[0], img.shape[1]), dtype=int)
        representation_score = representation_score.reshape((-1))
        representation_score_shared = multiprocessing.Array('i', representation_score)

        nb_jobs = 4
        step = math.ceil(img.shape[0]/nb_jobs)
        jobs = []
        increment = 0
        for job in range(nb_jobs):
            start = increment
            finish = start+step if (start+step) <= img.shape[0] else img.shape[0]
            p = multiprocessing.Process(target=estimateRepresentation, args=(img[start:finish,:], distributions, representation_scores[-1][start:finish,:], start, finish, representation_score_shared))
            jobs.append(p)
            p.start()
            increment += step

        for job in jobs:
            job.join()

        representation_score[0:representation_score.shape[0]] = representation_score_shared[0:representation_score.shape[0]]
        representation_score = representation_score.reshape((img.shape[0], img.shape[1]))

        representation_scores.append(representation_score)
        
        print('Storing Representation Image...')
        cv2.imwrite(f'representation_score{iteration}.jpg', representation_score)


        first = False
        iteration += 1


    # Color & Alpha Minimization
    print('Minimizing Energy Function For Each Pixel...')
    representation_score = np.empty((img.shape[0], img.shape[1]))
    # TODO : Implement MultiProcessing
    # Multithreading
    nb_threads = 6
    step = math.ceil(img.shape[0]/nb_threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        increment = 0
        for index in range(nb_threads):
            start = increment
            finish = start+step if (start+step) <= img.shape[0] else img.shape[0]
            future = executor.submit(chunk_minimize_function, img[start:finish,:], alpha_layers, color_layers)
            res = future.result()
            representation_score[start:finish,:] = res[2]
            for index in range(len(alpha_layers)):
                alpha_layers[index][start:finish,:] = res[0]
                color_layers[index][start:finish,:] = res[1]
            increment += step

    #TODO : verify feasability
    for index in range(len(color_layers)):
        layer = np.empty((img.shape[0], img.shape[1], 4))
        layer[:,:,0:3] = color_layers[index]
        layer[:,:,3] = alpha_layers[index]

        cv2.imwrite(f'layer_{index}.png', layer)