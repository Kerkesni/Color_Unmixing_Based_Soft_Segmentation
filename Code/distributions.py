import cv2
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
import argparse

# Global Variables
seeds = []
alpha_layers = []
color_layers = []
distributions = []
representation_scores = []
representation_threshold = 25

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

# Returns Pixels In a Bin
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
        # 21x21 Kernel Limits
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
def estimateDistribution(img, seed_coords, size=10):
    # Distribution Estimation
    lower_h = seed_coords[0]-size if seed_coords[0]-size >= 0 else 0
    higher_h = seed_coords[0]+size if seed_coords[0]-size < img.shape[0] else img.shape[0]-1

    lower_l = seed_coords[1]-size if seed_coords[1]-size >= 0 else 0
    higher_l = seed_coords[1]+size if seed_coords[1]+size < img.shape[1] else img.shape[1]-1

    patch = img[lower_h:higher_h, lower_l:higher_l].reshape(-1, 3)

    # Calculating mean and covariance matrix
    covariance_matrix, mean_vector = cv2.calcCovarMatrix(patch, None, cv2.COVAR_ROWS | cv2.COVAR_SCALE | cv2.COVAR_NORMAL)

    if np.sum(covariance_matrix) == 0:
        covariance_matrix = np.array([
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]
        ])
    else:
        # Inverting Covariance Matrix
        if np.linalg.cond(covariance_matrix) < 1/sys.float_info.epsilon:
            covariance_matrix = np.linalg.inv(covariance_matrix)
        else:
            covariance_matrix = np.linalg.pinv(covariance_matrix)

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
            # Calculating mahalanobis distance
            Di[j,i] = min(representation_score[j,i], mahalanobis(img[j,i], distribution[0], distribution[1]))
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

                    D1 = mahalanobis(u1, distributions[dist1][0], distributions[dist1][1])

                    D2 = mahalanobis(u2, distributions[dist2][0], distributions[dist2][1])
                    tmp_F = alpha1*D1 + alpha2*D2

                    rep_score[start+j,i] = min(tmp_F, rep_score[start+j,i], minDi[j,i])


if __name__ == '__main__':

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
    img_dim = (width, height)
    img = cv2.resize(img, img_dim)
    seeds_image = np.copy(img)

    cv2.imwrite('./res/original.jpg', img)

    # Calculating image gradient
    print('Calculating Gradient...')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray,cv2.CV_64F)

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

        cv2.circle(seeds_image, seed_coords, 5, (0, 255, 0), -1)
        cv2.imwrite(f'./res/seed{iteration}.jpg', seeds_image)

        first = False
        iteration += 1


    # save distributions, color_layers, alpha_layers
    print(f'Estimated {len(distributions)} distributions')
    print('Storing Distributions...')
    output = open("./data/dist.dat", "wb")
    pickle.dump(distributions, output)