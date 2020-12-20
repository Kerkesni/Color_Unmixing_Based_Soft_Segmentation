import multiprocessing
import cv2
import numpy as np
import math


# def f(a, b, index):
#     with shared_arr.get_lock():
#         b[start:finish,:] = a[start:finish,:]*2

# if __name__ == '__main__':

#     img = cv2.imread('original.jpg')
#     b = multiprocessing.Array('i', img)
#     step = math.ceil(img.shape[0]/nb_jobs)
#     nb_jobs = 2
#     jobs = []
#     increment = 0
#     for job in range(nb_jobs):
#         start = increment
#         finish = start+step if (start+step) <= img.shape[0] else img.shape[0]
#         p = multiprocessing.Process(target=f, args=(img, b, start, finish))
#         jobs.append(p)
#         p.start()
#         increment += step

#     for job in jobs:
#         job.join()
    
#     print(a[0,0])
#     print(b[0,0])

img = cv2.imread('original.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_flat = img_gray.reshape((-1))
print(img_flat.shape)
arr = multiprocessing.Array('i', img_flat)
arr[0:512000] = img_flat
# print(arr[0])
# print(img_flat[0])