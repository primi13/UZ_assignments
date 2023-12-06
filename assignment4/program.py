from a4_utils import *
import os
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


'''because np.floor(0.03 / 0.05) = 5, which is wrong, because 0.03 / 0.05
=5.999999999999999, because of awkward bit representation of numbers'''
def round_down(num):
    if abs(round(num) - num) < 0.000001:
        whole = round(num)
    else:
        whole = math.floor(num)
    return np.int32(whole)

def norm(np_arr):
    return np_arr / np.sum(np_arr)

def normalizeValues(arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    return (arr - min_val) / (max_val - min_val)

def imread(path):
    """
    Reads an image in RGB order. Image type is transformed from uint8 to float, and
    range of values is reduced from [0, 255] to [0, 1].
    """
    I = Image.open(path).convert('RGB')  # PIL image.
    I = np.asarray(I)  # Converting to Numpy array.
    I = I.astype(np.float64) / 255
    return I

def to_gray(image):
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    return (red + green + blue) / 3

# only puts values to 0 if they are bellow threshold
def normal_thresholding(image_gray, threshold):
    image_mask = np.copy(image_gray)
    image_mask[image_gray < threshold] = 0
    return image_mask

# puts values to 0 if they are bellow threshold and also puts them to 1
# if they are equal or greater than threshold
def normal_thresholding2(arr, tsh):
    arr_new = arr.copy()
    arr_new[arr_new < tsh] = 0
    arr_new[arr_new >= tsh] = 1
    return arr_new

around_pixel = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
def is_not_max_in_neighborhood(y, x, arr):
    r, c, *_ = arr.shape
    for i in range(8):
        y_inc, x_inc = around_pixel[i]
        y_new = y + y_inc
        x_new = x + x_inc
        # we cannot access a cell in the array if its index is over the edge
        if y_new < 0 or y_new >= r or x_new < 0 or x_new >= c:
            continue
        if arr[y_new, x_new] > arr[y, x]:
            return True
    return False

def nonmaxima_suppression_box(acc_arr):
    acc_arr_new = acc_arr.copy()
    r, c, *_ = acc_arr_new.shape
    for y in range(r):
        for x in range(c):
            if is_not_max_in_neighborhood(y, x, acc_arr_new):
                acc_arr_new[y, x] = 0
    return acc_arr_new


# 1

# 1a
    """
    Blobs are detected (Tak kot krogle luknje in vogali). 
    With a larger sigma larger blobs are detected.
    For example a Laplacian function looks like a blob.
    """
def first_derivatives(I, sigma):
    g = gauss(sigma)
    d = gaussdx(sigma)
    Ix = convolve(I, g.T, d)
    Iy = convolve(I, g, d.T)
    return Ix, Iy

def second_derivatives(I ,sigma):
    g = gauss(sigma)
    d = gaussdx(sigma)
    Ix, Iy = first_derivatives(I, sigma)
    Ixx = convolve(Ix, g.T, d) # add y
    Ixy = convolve(Ix, g, d.T) # add y
    Iyy = convolve(Iy, g, d.T) # add x
    return Ixx, Ixy, Iyy

def hessian_points(I, sigma):
    Ixx, Ixy, Iyy = second_derivatives(I, sigma)
    return (Ixx * Iyy) - (Ixy ** 2)

def case1a():
    graf_a = imread("data/graf/graf_a.jpg")
    graf_a_gray = to_gray(graf_a)
    _, axes = plt.subplots(2, 3)
    for i in [3, 6, 9]:
        graf_a_hessian_det = hessian_points(I=graf_a_gray, 
                                            sigma=i)
        index = int(i / 3) - 1
        axes[0, index].imshow(graf_a_hessian_det)
        axes[0, index].set_title(f"Sigma = {i}, no tsh or supp")
        graf_a_tsh = normal_thresholding(graf_a_hessian_det, 0.004)
        graf_a_maxima = nonmaxima_suppression_box_flexible(arr=graf_a_tsh,
                                                           kernel_size=51)

        axes[1, index].imshow(graf_a_gray, cmap="gray")
        y, x = np.where(graf_a_maxima > 0)
        axes[1, index].scatter(x, y, color="red", marker="x")
        axes[1, index].set_title(f"After tsh and supp")
    plt.show()


# 1b
"""
Yes, at both Hessian and Harris points appear on the same structures in the
image, although there are less Hessian points in the comparison. It seems like
Harris points appear also on corners, not only in blobs or holes.
"""

def nonmaxima_suppression_box_flexible(arr, kernel_size=3):
    assert kernel_size % 2 == 1  #it must be odd
    N = int((kernel_size - 1) / 2)
    arr_new = arr.copy()
    r, c, *_ = arr_new.shape
    for y in range(r):
        for x in range(c):
            if arr_new[y, x] == 0:
                continue
            if arr_new[y, x] < np.max(arr_new[max(y - N, 0):min(y + N, r),
                                               max(x - N, 0):min(x + N, c)]):
                arr_new[y, x] = 0
    return arr_new
            
def harris_points(I, sigma, sigma_tilde, alpha):
    Ix, Iy = first_derivatives(I, sigma)
    g = gauss(sigma_tilde)
    IxIxg = convolve(Ix * Ix, g, g.T)
    IxIyg = convolve(Ix * Iy, g, g.T)
    IyIyg = convolve(Iy * Iy, g, g.T)
    
    det = (IxIxg * IyIyg) - (IxIyg ** 2)
    trace = IxIxg + IyIyg
    return det - alpha * (trace ** 2)

def case1b():
    graf_a = imread("data/graf/graf_a.jpg")
    graf_a_gray = to_gray(graf_a)
    _, axes = plt.subplots(2, 3)
    for i in [3, 6, 9]:
        graf_a_harris_cond = harris_points(I=graf_a_gray, 
                                           sigma = i, 
                                           sigma_tilde=1.6 * i,
                                           alpha=0.06)
        index = int(i / 3) - 1
        axes[0, index].imshow(graf_a_harris_cond)
        axes[0, index].set_title(f"Sigma = {i}, no tsh or supp")
        graf_a_tsh = normal_thresholding(graf_a_harris_cond, 1.5e-6)
        graf_a_maxima = nonmaxima_suppression_box_flexible(arr=graf_a_tsh,
                                                           kernel_size=51)
        axes[1, index].imshow(graf_a_gray, cmap="gray")
        y, x = np.where(graf_a_maxima > 0)
        axes[1, index].scatter(x, y, color="red", marker="x")
        axes[1, index].set_title(f"After tsh and supp")
    plt.show()
    
    # comparing the Hessian and Harris algorithms results
    _, axes = plt.subplots(2, 3)
    for i in [3, 6, 9]:
        index = int(i / 3) - 1

        graf_a_hessian_det = hessian_points(I=graf_a_gray, 
                                            sigma=i)
        axes[0, index].imshow(graf_a_hessian_det)
        axes[0, index].set_title(f"Sigma = {i}, Hessian")

        graf_a_harris_cond = harris_points(I=graf_a_gray, 
                                           sigma = i, 
                                           sigma_tilde=1.6 * i,
                                           alpha=0.06)
        axes[1, index].imshow(graf_a_harris_cond)
        axes[1, index].set_title(f"Sigma = {i}, Harris")
    plt.show()

    # comparing the Hessian and Harris algorithms results on the images
    _, axes = plt.subplots(2, 3)
    for i in [3, 6, 9]:
        index = int(i / 3) - 1
        
        graf_a_hessian_det = hessian_points(I=graf_a_gray, 
                                            sigma=i)
        graf_a_tsh = normal_thresholding(graf_a_hessian_det, 0.004)
        graf_a_maxima = nonmaxima_suppression_box_flexible(arr=graf_a_tsh,
                                                           kernel_size=51)
        axes[0, index].imshow(graf_a_gray, cmap="gray")
        y, x = np.where(graf_a_maxima > 0)
        axes[0, index].scatter(x, y, color="red", marker="x")
        axes[0, index].set_title(f"Sigma = {i}, Hessian")
        
        graf_a_harris_cond = harris_points(I=graf_a_gray, 
                                           sigma = i, 
                                           sigma_tilde=1.6 * i,
                                           alpha=0.06)
        graf_a_tsh = normal_thresholding(graf_a_harris_cond, 1.5e-6)
        graf_a_maxima = nonmaxima_suppression_box_flexible(arr=graf_a_tsh,
                                                           kernel_size=51)
        axes[1, index].imshow(graf_a_gray, cmap="gray")
        y, x = np.where(graf_a_maxima > 0)
        axes[1, index].scatter(x, y, color="red", marker="x")
        axes[1, index].set_title(f"Sigma = {i}, Harris")
    plt.show()



# 2

# 2a
def whole_Hessian(I_gray, sigma = 3, threshold = 0.004, kernel_size = 51):
    graf_a_hessian_det = hessian_points(I_gray, sigma)
    graf_a_tsh = normal_thresholding(graf_a_hessian_det, threshold)
    return nonmaxima_suppression_box_flexible(arr=graf_a_tsh,
                                                           kernel_size=kernel_size)

def whole_Harris(I_gray, sigma = 3, sigma_tilde = None, aplha = 0.06, 
                 threshold = 1e-6, kernel_size = 51):
        if sigma_tilde is None:
            sigma_tilde = 1.6 * sigma
        graf_a_harris_cond = harris_points(I=I_gray, 
                                           sigma = sigma, 
                                           sigma_tilde=sigma_tilde,
                                           alpha=aplha)
        graf_a_tsh = normal_thresholding(graf_a_harris_cond, threshold)
        return nonmaxima_suppression_box_flexible(arr=graf_a_tsh,
                                                           kernel_size=kernel_size)
        
# this is hellinger distance between two histograms
def hellinger(h1, h2):
    sqrt1 = np.sqrt(h1)
    sqrt2 = np.sqrt(h2)
    diff_sqrt = sqrt1 - sqrt2
    square = diff_sqrt * diff_sqrt
    value_sumed = np.sum(square)
    value_half = value_sumed / 2
    return np.sqrt(value_half)

def find_correspondences(desc1, desc2):
    correspondences = []
    len1 = desc1.shape[0]
    len2 = desc2.shape[0]
    for i in range(len1):
        min = np.inf
        min_index = None
        for j in range(len2):
            dist = hellinger(desc1[i], desc2[j])
            if dist < min:
                min = dist
                min_index = j
        correspondences.append((i, min_index))
    return correspondences

def align_feature_points_coordinates(arr, corres):
    aligned_arr = []
    for _, j in corres:
        aligned_arr.append(arr[j])
    return aligned_arr

def case2a():
    """
    Simple descriptors:
    Comparison gets worse for 8 bins than it is for 16.
    And then it doesn't get any better for more than 16 bins, for 50 bins
    the correspondeces are worse.
    
    Radius 10 gives worse results than 40. Between 40 and 90 it is pretty much the same
    and after 100 it gets worse.
    
    Sigma 1 and 6 give good result, however 3 doesn't give good results, interesting, it's
    probably just a coincidence.
    """
    graf_a_small = imread("data/graf/graf_a_small.jpg")
    graf_a_gray = to_gray(graf_a_small)
    graf_a_harris = whole_Harris(graf_a_gray, sigma=3, kernel_size = 31)
    graf_a_y, graf_a_x = np.where(graf_a_harris > 0)
    desc1 = simple_descriptors(I=graf_a_gray, Y=graf_a_y, X=graf_a_x, sigma=1)
    graf_a_feature_points_coordinates = list(zip(graf_a_x, graf_a_y))
    
    graf_b_small = imread("data/graf/graf_b_small.jpg")
    graf_b_gray = to_gray(graf_b_small)
    graf_b_harris = whole_Harris(graf_b_gray, sigma=3, kernel_size = 31)
    graf_b_y, graf_b_x = np.where(graf_b_harris > 0)
    desc2 = simple_descriptors(I=graf_b_gray, Y=graf_b_y, X=graf_b_x, sigma=1)
    graf_b_feature_points_coordinates = list(zip(graf_b_x, graf_b_y))
    
    correspondences = find_correspondences(desc1, desc2)
    graf_b_feature_points_coordinates = align_feature_points_coordinates(graf_b_feature_points_coordinates, 
                                                                         correspondences)
    
    display_matches(graf_a_gray, graf_a_feature_points_coordinates, 
                    graf_b_gray, graf_b_feature_points_coordinates)
    
    
# 2b
"""
Matches are more accurate than before,
there are definitely less outliers.
"""
def retain_only_symmetric_correspondences(cor1, cor2):
    correspondences = []
    len1 = len(cor1)
    len2 = len(cor2)
    for i in range(len1):
        if cor2[cor1[i][1]][1] == i:
            correspondences.append(cor1[i])
    return correspondences

def retain_only_pts_that_matched(pts1, pts2, cor):
    pts1_new = []
    pts2_new = []
    for i, j in cor:
        pts1_new.append(pts1[i])
        pts2_new.append(pts2[j])
    return pts1_new, pts2_new

def find_matches(I1_gray, I2_gray, tsh=None):
    I1_harris = whole_Harris(I1_gray, sigma=3, kernel_size = 31)
    I1_y, I1_x = np.where(I1_harris > 0)
    desc1 = simple_descriptors(I=I1_gray, Y=I1_y, X=I1_x, sigma=1)
    I1_feature_pts_coord = list(zip(I1_x, I1_y))
    
    I2_harris = whole_Harris(I2_gray, sigma=3, kernel_size = 31)
    I2_y, I2_x = np.where(I2_harris > 0)
    desc2 = simple_descriptors(I=I2_gray, Y=I2_y, X=I2_x, sigma=1)
    I2_feature_pts_coord = list(zip(I2_x, I2_y))
    
    correspondences = None
    if tsh == None:
        correspondences_from_1_to_2 = find_correspondences(desc1, desc2)
        correspondences_from_2_to_1 = find_correspondences(desc2, desc1)
        correspondences = retain_only_symmetric_correspondences(
                            correspondences_from_1_to_2, 
                            correspondences_from_2_to_1)
    else:
        correspondences_from_1_to_2 = find_correspondences2(desc1, desc2, tsh)
        correspondences_from_2_to_1 = find_correspondences2(desc2, desc1, tsh)
        correspondences = retain_only_symmetric_correspondences2(
                            correspondences_from_1_to_2, 
                            correspondences_from_2_to_1)
    print(correspondences)
    print()

    I1_feature_pts_coord, I2_feature_pts_coord = retain_only_pts_that_matched(
                                                        I1_feature_pts_coord,
                                                        I2_feature_pts_coord, 
                                                        correspondences)
    return I1_feature_pts_coord, I2_feature_pts_coord

def case2b():
    graf_a_gray = to_gray(imread("data/graf/graf_a_small.jpg"))

    graf_b_gray = to_gray(imread("data/graf/graf_b_small.jpg"))

    graf_a_feature_pts_coord, graf_b_feature_pts_coord = find_matches(
                                                            graf_a_gray, 
                                                            graf_b_gray)
    display_matches(graf_a_gray, graf_a_feature_pts_coord,
                    graf_b_gray, graf_b_feature_pts_coord)


# 2c
"""
Strategy 3: Calculate the distance between point A and the most similar keypoint and
then also the distance between A and the second-most similar keypoint in the other image.
The ration of these two distances will be low for distinctive key-points and high for
non-distinctive ones. Threshold approximately 0.8 gives good results with SIFT.

It put threshold on 0.9 and it works really good, almost all pairs of
feature points are perfects matches.
"""
def retain_only_symmetric_correspondences2(cor1, cor2):
    correspondences = []
    len1 = len(cor1)
    len2 = len(cor2)
    for i in range(len1):
        pt_in_right_image = cor1[i][1]
        for j in range(len2):
            if cor2[j][0] == pt_in_right_image and cor2[j][1] == cor1[i][0]:
                correspondences.append(cor1[i])
    return correspondences


def find_correspondences2(desc1, desc2, tsh):
    correspondences = []
    len1 = desc1.shape[0]
    len2 = desc2.shape[0]
    for i in range(len1):
        min = np.inf
        second_min = min
        min_index = None
        for j in range(len2):
            dist = hellinger(desc1[i], desc2[j])
            if dist < min:
                second_min = min
                min = dist
                min_index = j
            elif dist < second_min:
                second_min = dist
        if min / second_min < tsh:
            correspondences.append((i, min_index))
    return correspondences


def case2c():
    graf_a_gray = to_gray(imread("data/graf/graf_a_small.jpg"))

    graf_b_gray = to_gray(imread("data/graf/graf_b_small.jpg"))

    graf_a_feature_pts_coord, graf_b_feature_pts_coord = find_matches(
                                                            graf_a_gray, 
                                                            graf_b_gray, 
                                                            tsh=0.9)
    display_matches(graf_a_gray, graf_a_feature_pts_coord,
                    graf_b_gray, graf_b_feature_pts_coord)

#case1a()
#case1b()

#case2a()
case2b()
case2c()