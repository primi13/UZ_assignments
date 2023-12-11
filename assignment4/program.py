from a4_utils import *
import os
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import random


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

def find_matches(I1, I2, tsh=None, sigma=3, kernel_size=31, 
                 sigma_desc=1, sift=False):
    I1_gray = to_gray(I1)
    I2_gray = to_gray(I2)

    I1_harris = whole_Harris(I1_gray, sigma=sigma, kernel_size=kernel_size)
    I1_y, I1_x = np.where(I1_harris > 0)
    desc1 = None
    if sift:
        desc1 = descriptor_SIFT(I=I1_gray, Y=I1_y, X=I1_x, sigma=sigma_desc)
        make_rotation_invariant(desc1)
    else:
        desc1 = simple_descriptors(I=I1_gray, Y=I1_y, X=I1_x, sigma=sigma_desc)
    I1_feature_pts_coord = list(zip(I1_x, I1_y))
    
    I2_harris = whole_Harris(I2_gray, sigma=sigma, kernel_size=kernel_size)
    I2_y, I2_x = np.where(I2_harris > 0)
    desc2 = None
    if sift:
        desc2 = descriptor_SIFT(I=I2_gray, Y=I2_y, X=I2_x, sigma=sigma_desc)
        make_rotation_invariant(desc2)
    else:
        desc2 = simple_descriptors(I=I2_gray, Y=I2_y, X=I2_x, sigma=sigma_desc)
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

    I1_feature_pts_coord, I2_feature_pts_coord = retain_only_pts_that_matched(
                                                        I1_feature_pts_coord,
                                                        I2_feature_pts_coord, 
                                                        correspondences)
    return I1_feature_pts_coord, I2_feature_pts_coord

def find_and_display_matches(I1_path, I2_path, tsh=None):
    I1 = imread(I1_path)
    I2 = imread(I2_path)

    I1_pts, I2_pts = find_matches(I1, I2, tsh=tsh)
    display_matches(I1, I1_pts,
                    I2, I2_pts)

def case2b():
    find_and_display_matches("data/graf/graf_a_small.jpg", 
                             "data/graf/graf_b_small.jpg")

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
    find_and_display_matches("data/graf/graf_a_small.jpg", 
                             "data/graf/graf_b_small.jpg",
                             tsh=0.9)

# 2d
def doG(I, sigma=1):
    laplace_piramid = []
    g = gauss(sigma)
    for _ in range(4):
        for _ in range(4):
            Ib = convolve(I, g, g.T)
            #Is = Ib[::2, ::2]
            laplace_piramid.append(Ib - I)
            I = Ib
        Is = Ib[::2, ::2]
        I = Is
    return laplace_piramid
        
def angle_to_index(angle):
    angle = (angle + (9*np.pi)/8) % (2 * np.pi)
    # 0 is left, 1 is left down and so on around the circle
    return (np.floor(angle / (np.pi / 4))).astype(int)

def coord_to_grid_index(x, minx, w, y, miny, h):
    return int(math.floor((x - minx)/w)), int(math.floor((y - miny)/h))


def descriptor_SIFT(I, Y, X, n_bins = 8, radius = 40, sigma = 2):
    """
    Computes descriptors for locations given in X and Y.

    I: Image in grayscale.
    Y: list of Y coordinates of locations. (Y: index of row from top to bottom)
    X: list of X coordinates of locations. (X: index of column from left to right)

    Returns: tensor of shape (len(X), 4^2, n_bins^2), so for each point a feature of 
    length 16 * n_bins.
    """
    assert np.max(I) <= 1, "Image needs to be in range [0, 1]"
    assert I.dtype == np.float64, "Image needs to be in np.float64"
    assert radius % 4 == 0, "Radius needs to be a multiple of 4, so the split into \
    a grid of 4x4 is nice and simple."

    # if radius = 40, then 2*radius = 80
    # 6sigma + 1 = 2*radius => sigma = ((2*radius)-1)/6
    w = gauss(1/3 * radius) # made a little bigger than it needs to be just in case
    w2d = np.outer(w, w)
    height, width = w2d.shape
    w_use = w2d[int(height/2)-radius:int(height/2)+radius, 
                int(width/2)-radius:int(width/2)+radius]
    
    g = gauss(sigma)
    d = gaussdx(sigma)

    Ix = convolve(I, g.T, d)
    Iy = convolve(I, g, d.T)

    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    mag = np.floor(mag * ((n_bins - 1) / np.max(mag)))

    ang = np.arctan2(Iy, Ix)
    ang = np.floor(ang * ((n_bins - 1) / np.max(ang)))

    desc = []
    for y, x in zip(Y, X):
        miny = max(y - radius, 0)
        maxy = min(y + radius, I.shape[0])
        minx = max(x - radius, 0)
        maxx = min(x + radius, I.shape[1])
        w = (maxx - minx) / 4
        h = (maxy - miny) / 4
        
        a = np.zeros((4, 4, n_bins))
        for i in range(miny, maxy):
            for j in range(minx, maxx):
                a[*coord_to_grid_index(j, minx, w, i, miny, h), 
                  angle_to_index(ang[i, j])] += mag[i, j] * w_use[i - miny, j - minx]

        a = a.reshape(-1)
        a /= np.sum(a)

        desc.append(a)

    return np.array(desc)

def make_rotation_invariant(histograms):
    for i, histogram in enumerate(histograms):
        max = -math.inf
        max_index = None
        for j, val in enumerate(histogram):
            if val > max:
                max = val
                max_index = j
        histograms[i] = np.concatenate([histogram[max_index:],histogram[:max_index]])

def case2d():
    graf_a = imread("data/graf/graf_a.jpg")
    graf_b = imread("data/graf/graf_b.jpg")
    pts1, pts2 = find_matches(graf_a, graf_b, sift=True, tsh=0.5)
    display_matches(graf_a, pts1, graf_b, pts2)


# 2e
def case2e():
    sift = cv2.SIFT_create()
    cap = cv2.VideoCapture("data/my_video.mp4")
    ret, frame = cap.read()
    while(1):
        ret, frame = cap.read()
        keypoints = sift.detectAndCompute(frame, None)
        output_frame = cv2.drawKeypoints(frame, keypoints, 0, (0, 0, 255), 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
        cv2.imshow('frame', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False:
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('frame', output_frame)
        
        
# 3
"""
p1 is for scaling, p2 is for rotating and p3 and p4 are for translating.

1. Define the set of corresponding points.
2. Define the linear transformation.
3. Define the per-point error and combine all results into a vector e.
4. Rewrite the error into a form e = Ap - b
5. Solve by pseudoinverse p = (A.T * A)^(-1) * A.T * b or 
p = (A.T * W * A)^(-1) * A.T * W * b, if you also have weights for errors
"""

# 3a
def file_to_pts(arr2d):
    pts1 = []
    pts2 = []
    for x1, y1, x2, y2 in arr2d:
        pts1.append((x1, y1))
        pts2.append((x2, y2))
    return pts1, pts2

def constructA(pts1, pts2):
    A = []
    for p1, p2 in zip(pts1, pts2):
        xr = p1[0]
        yr = p1[1]
        xt = p2[0]
        yt = p2[1]
        A.append(np.array([xr, yr, 1, 0, 0, 0, -xt*xr, -xt*yr, -xt]))
        A.append(np.array([0, 0, 0, xr, yr, 1, -yt*xr, -yt*yr, -yt]))
    return np.array(A)

def estimate_homography(pts1, pts2):
    A = constructA(pts1, pts2)
    _, _, VT = np.linalg.svd(A)
    V = VT.T
    h = V[:,-1] / V[-1,-1]
    return h.reshape(3, 3)

def transform_image_plane(I1_path, I2_path, name1, name2, pts_path, 
                          my_warp=False):
    I1 = imread(I1_path)
    I2 = imread(I2_path)
    I1_gray = to_gray(I1)
    I2_gray = to_gray(I2)
    
    pts = np.loadtxt(pts_path)
    pts1, pts2 = file_to_pts(pts)
    
    display_matches(I1, pts1, I2, pts2)
    H = estimate_homography(pts1, pts2)
    print(H)
    
    I1_changed = None
    if my_warp:
        # I used the inverse of matrix H
        I1_changed = my_warp_perspective(I1, np.linalg.inv(H), (I2.shape[1], I2.shape[0]))
    else:
        I1_changed = cv2.warpPerspective(I1, H, (I2.shape[1], I2.shape[0]))
    _, axes = plt.subplots(1, 3)
    axes[0].imshow(I1)
    if name1 is not None:
        axes[0].set_title(name1)
    axes[1].imshow(I1_changed)
    if name1 is not None:
        axes[1].set_title(f"{name1}_changed")
    axes[2].imshow(I2)
    if name2 is not None:
        axes[2].set_title(name2)
    plt.show()

def case3a():
    transform_image_plane("data/newyork/newyork_a.jpg", 
                          "data/newyork/newyork_b.jpg",
                          "newyork_a",
                          "newyork_b",
                          "data/newyork/newyork.txt")
    transform_image_plane("data/graf/graf_a.jpg", 
                          "data/graf/graf_b.jpg", 
                          "graf_a",
                          "graf_b",
                          "data/graf/graf.txt")
    

# 3b
def euclidean(h1, h2):
    diff = h1 - h2
    square = diff * diff
    value_sumed = np.sum(square)
    return np.sqrt(value_sumed)

def RANSAC(path1, path2,  name1=None, name2=None, sigma=3, sift_tsh=None, 
           kernel_size=31, k=10, ransac_tsh=10, sift=False):
    I1 = imread(path1)
    I2 = imread(path2)
    pts1, pts2 = find_matches(I1, I2, 
                              tsh=sift_tsh,
                              sigma=sigma, 
                              kernel_size=kernel_size,
                              sift=sift)
    display_matches(I1, pts1, I2, pts2)
    print(pts1)
    print()
    print(pts2)
    print()
    num_pts = len(pts1)
    
    # RANSAC loop
    best_H = None
    max_outliers = 0
    for _ in range(k):
        # 1. select 4 random correspondences
        pts1_selected = []
        pts2_selected = []
        for _ in range(4):
            idx = random.randint(0, num_pts - 1)
            pts1_selected.append(pts1[idx])
            pts2_selected.append(pts2[idx])
            
        # 2. estimate homography on those 4 random correspondences
        H = estimate_homography(pts1_selected, pts2_selected)
        
        # 3. project all other correspondences and check the number of inliers
        inliers = 0
        for p1, p2 in zip(pts1, pts2):
            p1_arr = np.array([p1[0], p2[1], 1])
            p1_transf_homog = np.matmul(H, p1_arr)
            p1_arr_transf = np.array([p1_transf_homog[0], p1_transf_homog[1]])
            p2_arr = np.array([p2[0], p2[1]])
            dist = euclidean(p1_arr_transf, p2_arr)
            if dist < ransac_tsh:
                inliers += 1
                
        # 4. maximize the number of inliers and remember the correspondences
        # when the number of inliers was maximal
        if inliers > max_outliers:
            max_outliers = inliers
            best_H = H

    print(best_H)
    print(max_outliers)

    # display the result of the ransac loop
    I1_changed = cv2.warpPerspective(I1, best_H, (I2.shape[1], I2.shape[0]))
    _, axes = plt.subplots(1, 3)
    axes[0].imshow(I1)
    if name1 is not None:
        axes[0].set_title(name1)
    axes[1].imshow(I1_changed)
    if name1 is not None:
        axes[1].set_title(f"{name1}_changed")
    axes[2].imshow(I2)
    if name2 is not None:
        axes[2].set_title(name2)
    plt.show()


def case3b():
    #RANSAC("data/graf/graf_a.jpg", "data/graf/graf_b.jpg", 
    #       name1="graf_a", name2="graf_b", k=1000,
    #       ransac_tsh=5, sift_tsh=0.4, sift=True, kernel_size=51)

    RANSAC("data/newyork/newyork_a.jpg", "data/newyork/newyork_b.jpg", 
           name1="newyork_a", name2="newyork_b", sigma=1, ransac_tsh=5, 
           k=1000, kernel_size=11)

# 3d
def my_warp_perspective(I, H, shape):
    changed_image = np.zeros((shape[1], shape[0], 3))
    for y in range(shape[0]):
        for x in range(shape[1]):
            divider = (H[2, 0]*x + H[2, 1]*y + 1)
            changed_x = int((H[0, 0]*x + H[0, 1]*y + H[0, 2]) / divider)
            changed_y = int((H[1, 0]*x + H[1, 1]*y + H[1, 2]) / divider)
            if changed_y < I.shape[0] and changed_y > 0 and \
                changed_x < I.shape[1] and changed_x > 0:
                changed_image[y, x] = I[changed_y, changed_x]
    return changed_image

def case3d():
    transform_image_plane("data/newyork/newyork_a.jpg", 
                        "data/newyork/newyork_b.jpg",
                        "newyork_a",
                        "newyork_b",
                        "data/newyork/newyork.txt", 
                        my_warp=True)




#case1a()
#case1b()

#case2a()
#case2b()
#case2c()
#case2d()
#case2e()

#case3a()
case3b()
#case3d()



""" lena = to_gray(imread("../assignment2/images/lena.png"))
laplace_piramid = doG(lena, sigma=3)
h = 4
w = 4
_, axes = plt.subplots(h, w)
for i in range(h):
    for j in range(w):
        index = i * w + j
        if index < len(laplace_piramid):
            axes[i, j].imshow(laplace_piramid[index], cmap="gray")
plt.show()
 """