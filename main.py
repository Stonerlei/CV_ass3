# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:42:03 2020

@author: Stoner
"""

import cv2

import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np


import math
 

# matplotlib inline

# read images

img1 = cv2.imread('e.png')

img2 = cv2.imread('d.png')
# h_bes = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype='float64')
# img2 = cv2.warpPerspective(img1,h_bes,(1000,1000))



img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


#sift

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)

matches = sorted(matches, key=lambda x:x.distance)


# img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:1], img2, flags=2)

# cv2.imshow('img',img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# get corresponding coordinates
points1f = cv2.KeyPoint_convert(keypoints_1)
points2f = cv2.KeyPoint_convert(keypoints_2)
ori_coord = np.zeros([2,0])
des_coord = np.zeros([2,0])

for match_points in matches[:200]:
    a = points1f[match_points.queryIdx]
    a = np.reshape(a, [2,1])
    b = points2f[match_points.trainIdx]
    b = np.reshape(b,[2,1])
    ori_coord = np.append(ori_coord,a,axis=1)
    des_coord = np.append(des_coord,b,axis=1)

# RANSAC
num_iter = 1000
iter_id = 0
lens = np.shape(ori_coord)
lens = lens[1]
one = np.ones([1,lens])
ori_coord = np.append(ori_coord, one, axis=0)
des_coord = np.append(des_coord, one, axis=0)
des_pred = np.zeros(np.shape(ori_coord))
num_inlier = 0
inlier_index = []
threshold = 5
num_sample = 4
while iter_id < num_iter:
    iter_id += 1
    # generate random index
    A = np.zeros([2*num_sample,9], dtype='float64')
    samples = np.random.randint(0,lens,size=(num_sample,1))
    # construct matrix A
    for i in range(num_sample):
        A[0+2*i] = np.array([ori_coord[0][samples[i]], ori_coord[1][samples[i]], 1, 0, 0, 0, -des_coord[0][samples[i]]*ori_coord[0][samples[i]], -des_coord[0][samples[i]]*ori_coord[1][samples[i]], -des_coord[0][samples[i]]])
        A[1+2*i] = np.array([0, 0, 0, ori_coord[0][samples[i]], ori_coord[1][samples[i]], 1, -des_coord[1][samples[i]]*ori_coord[0][samples[i]], -des_coord[1][samples[i]]*ori_coord[1][samples[i]], -des_coord[1][samples[i]]])
    
    print(iter_id)
    # calculate matrix h
    u,s,v = np.linalg.svd(A)
    v = np.transpose(v)
    h = v[:,8]/v[8][8]
    h = np.reshape(h, [3,3])
    temp = 0
    temp_index = []
    for n in range(lens):
        ori = np.reshape(ori_coord[:,n],[3,1])
        des_pred[:,n] = np.matmul(h,ori_coord[:,n])
        des_pred[:,n] /= des_pred[2][n]
        dist = np.linalg.norm(des_coord[:,n]-des_pred[:,n])
        print(dist)
        if dist < threshold:
            temp += 1
            temp_index.append(n)
    if temp > num_inlier:
        num_inlier = temp
        inlier_index = temp_index
        h_best = h
        wn = math.pow(num_inlier/lens, num_sample)
        num_iter = int(math.log(0.01, 1-wn))               


# ordinary least squares
for iter_id in range(100):
    A = np.zeros([2*num_inlier,9], dtype='float64')
    for i in range(num_inlier):
         A[0+2*i] = np.array([ori_coord[0][inlier_index[i]], ori_coord[1][inlier_index[i]], 1, 0, 0, 0, -des_coord[0][inlier_index[i]]*ori_coord[0][inlier_index[i]], -des_coord[0][inlier_index[i]]*ori_coord[1][inlier_index[i]], -des_coord[0][inlier_index[i]]])
         A[1+2*i] = np.array([0, 0, 0, ori_coord[0][inlier_index[i]], ori_coord[1][inlier_index[i]], 1, -des_coord[1][inlier_index[i]]*ori_coord[0][inlier_index[i]], -des_coord[1][inlier_index[i]]*ori_coord[1][inlier_index[i]], -des_coord[1][inlier_index[i]]])
    # h_best = np.matmul(np.transpose(A),A)
    # h_best = np.linalg.inv(h_best)
    # h_best = np.matmul(h_best, np.transpose(A))
    
    # calculate matrix h
    u,s,v = np.linalg.svd(A)
    v = np.transpose(v)
    h = v[:,8]/v[8][8]
    h = np.reshape(h, [3,3])
    inlier_index = []
    num_inlier = 0
    for n in range(lens):
        ori = np.reshape(ori_coord[:,n],[3,1])
        des_pred[:,n] = np.matmul(h,ori_coord[:,n])
        des_pred[:,n] /= des_pred[2][n]
        dist = np.linalg.norm(des_coord[:,n]-des_pred[:,n])
        print(dist)
        if dist < threshold:
            num_inlier += 1
            inlier_index.append(n)

     

# pred = ndimage.affine_transform(img2,h[:2,:2],(h[0,2],h[1,2]))
img1 = plt.imread('e.png')
dst = cv2.warpPerspective(img1,h,(1000,1000))
plt.imshow(dst)
plt.show()
plt.pause(0)

plt.imshow(img2,cmap='gray')
plt.show()
plt.pause(0)