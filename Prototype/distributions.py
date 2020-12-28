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
import pickle

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

# Translate vector to range
def translate_vec(v, leftMin=0, leftMax=1, rightMin=0, rightMax=255):
    tmp = np.empty_like(v)
    for index in range(len(v)):
        tmp[index] = translate(v[index], leftMin, leftMax, rightMin, rightMax)
    return tmp

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
def getSeedPixel(img, votes, winner_bin, coords, gradient, size=10):
    # Getting The Next Seed Pixel
    scores = []
    for pixel in coords:
        # 20x20 Kernel Limits
        lower_h = pixel[0]-size if pixel[0]-size >= 0 else 0 
        higher_h = pixel[0]+size if pixel[0]+size < img.shape[0] else img.shape[0]-1

        lower_l = pixel[1]-size if pixel[1]-size >= 0 else 0
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
def estimateDistribution(img, seed_coords, size=3):
    # Distribution Estimation
    lower_h = seed_coords[0]-size if seed_coords[0]-size >= 0 else 0
    higher_h = seed_coords[0]+size if seed_coords[0]-size < img.shape[0] else img.shape[0]-1

    lower_l = seed_coords[1]-size if seed_coords[1]-size >= 0 else 0
    higher_l = seed_coords[1]+size if seed_coords[1]+size < img.shape[1] else img.shape[1]-1

    patch = img[lower_h:higher_h, lower_l:higher_l]
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
            if np.sum(inv_cov) == 0:
                inv_cov = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
            Di[j,i] = min(representation_score[j,i], mahalanobis(img[j,i], distribution[0], inv_cov))
    return Di

# Estimating representation score
def estimateRepresentation(img, distributions, representation_score, start, finish, dim, representation_threshold=representation_threshold):

    rep_score = np.frombuffer(representation_score.get_obj())
    rep_score = rep_score.reshape(dim)

    # Getting the minimum mahalanobis distance
    minDi = getMahalanobis(img, distributions[-1], rep_score[start:finish, :])
    if len(distributions) < 2 :
        rep_score[start:finish, :] = minDi[:,:]
        return

    for dist1 in range(len(distributions)):
        for dist2 in range(len(distributions)):   
            if dist1 == dist2 :
                continue

            n =  distributions[dist1][0] - distributions[dist2][0]
            for j in range(img.shape[0]):
                for i in range(img.shape[1]):

                    if(rep_score[start+j,i] < representation_threshold):
                        continue

                    u1 = img[j,i] - (((img[j,i] - distributions[dist1][0])*n)/n*n)*n
                    u2 = img[j,i] - (((img[j,i] - distributions[dist2][0])*n)/n*n)*n

                    alpha1 = np.linalg.norm(img[j,i] - u1)/np.linalg.norm(u1-u2)
                    alpha2 = 1 - alpha1

                    if np.linalg.cond(distributions[dist1][1]) < 1/sys.float_info.epsilon:
                                inv_cov = np.linalg.inv(distributions[dist1][1])
                    else:
                        inv_cov = distributions[dist1][1]

                    if np.sum(inv_cov) == 0:
                        inv_cov = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

                    D1 = mahalanobis(u1, distributions[dist1][0], inv_cov)

                    if np.linalg.cond(distributions[dist2][1]) < 1/sys.float_info.epsilon:
                                inv_cov = np.linalg.inv(distributions[dist2][1])
                    else:
                        inv_cov = distributions[dist2][1]

                    if np.sum(inv_cov) == 0:
                        inv_cov = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

                    D2 = mahalanobis(u2, distributions[dist2][0], inv_cov)
                    tmp_F = alpha1*D1 + alpha2*D2

                    rep_score[start+j,i] = min(tmp_F, rep_score[start+j,i], minDi[j,i])


if __name__ == '__main__':

    # Reading Image
    print('Reading Image...')
    img = cv2.imread('../assets/e.jpg')
    percent = 10
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    img_dim = (width, height)
    img = cv2.resize(img, img_dim)

    cv2.imwrite('./res/original.jpg', img)

    # Calculating image gradient
    print('Calculating Gradient...')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray,cv2.CV_64F)

    # Initialization
    # print('Translating image from [0, 255] to [0, 1]')
    # img = img.astype('float') 
    # for j in range(img.shape[0]):
    #     for i in range(img.shape[1]):
    #         img[j,i] = [translate(img[j,i][0]), translate(img[j,i][1]), translate(img[j,i][2])]

    print('Initializing Layers...')
    representation_scores.append(np.full((img.shape[0], img.shape[1], 1), 255, dtype=int))

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
        seed_coords = getSeedPixel(img, votes, winner_bin, coords, laplacian)
        seeds.append(seed_coords)
        print(f'Seed Pixel Coords : {seed_coords}')

        # Getting Distribution Parameters
        print('Estimating Distribution...')
        distribution = estimateDistribution(img, seed_coords)
        distributions.append(distribution)
        print(f'Distribution : {distribution}')
    
        # Show distribution
        dist = np.full((100, 100, 3), distribution[0])
        cv2.imwrite(f'./res/dist_{iteration}.jpg', dist)
        del dist

        # Estimate Representation score
        print("Estimating representation scores...")
        # Multiprocessing
        representation_score = np.copy(representation_scores[-1])
        rep_dim = (representation_score.shape[0], representation_score.shape[1])
        representation_score = representation_score.flatten()
        representation_score_shared = multiprocessing.Array('d', representation_score)

        nb_jobs = 1
        step = math.ceil(img.shape[0]/nb_jobs)
        jobs = []
        increment = 0
        for job in range(nb_jobs):
            start = increment
            finish = start+step if (start+step) <= img.shape[0] else img.shape[0]
            p = multiprocessing.Process(target=estimateRepresentation, args=(img[start:finish,:], distributions, representation_score_shared, start, finish, rep_dim))
            jobs.append(p)
            p.start()
            increment += step

        for job in jobs:
            job.join()

        representation_score = np.copy(representation_score_shared.get_obj())
        representation_score = representation_score.reshape(rep_dim)
        representation_scores.append(representation_score)
        
        cv2.imwrite(f'./res/representation_score{iteration}.jpg', representation_score)


        first = False
        iteration += 1


    # save distributions, color_layers, alpha_layers
    print(f'Estimated {len(distributions)} distributions')
    print('Storing Distributions...')
    output = open("./data/dist.dat", "wb")
    pickle.dump(distributions, output)