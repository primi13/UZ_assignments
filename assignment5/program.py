from a5_utils import *
from a4_utils import *
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import random


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

def draw_epiline(l,h,w, axis):
	# l: line equation (vector of size 3)
	# h: image height
	# w: image width

	x0, y0 = map(int, [0, -l[2]/l[1]])
	x1, y1 = map(int, [w-1, -(l[2]+l[0]*w)/l[1]])

	axis.plot([x0,x1],[y0,y1],'r')
 
	axis.set_ylim([0,h])
	axis.invert_yaxis()


# 1

# 1b
def case1b():
    # graph of disparities for pz ranging from 2.5 (=f) and 200 with a step of 0.1
    f = 2.5#mm
    T = 120#mm
    disparities = []
    pzs = np.arange(2.5, 200, 0.1)
    for pz in pzs:
        disparities.append(f * T / pz)
    plt.plot(pzs, disparities)
    plt.title("Disparity in relation to pz:")
    plt.xlabel("pz [mm]")
    plt.ylabel("disparity [mm]")
    plt.show()
    

# 1d
def NCC(X, Y, x_avg, x_var):
    y_avg = np.average(Y)
    num = np.sum((X - x_avg) * (Y - y_avg))
    denom = np.sqrt(x_var * np.sum((Y - y_avg)**2))
    if denom == 0:
        return None
    return num / denom

def get_avg(matrix2D):
    r, c = matrix2D.shape
    sum = 0
    count = r * c
    for i in range(r):
        for j in range(c):
            sum += matrix2D[i, j]
    return sum / count

def get_variance(matrix2D, avg):
    r, *c = matrix2D.shape
    sum = 0
    for i in range(r):
        for j in range(c):
            sum += (matrix2D[i, j] - avg)**2
    return sum

def NCC2(X, Y, x_avg, x_var):
    y_avg = get_avg(Y)
    r, c = Y.shape

    # calculate numerator and (y_variance without dividing) together
    # in one double loop
    num = 0
    y_var = 0
    for i in range(r):
        for j in range(c):
            y_diff = Y[i, j] - y_avg
            num += (X[i, j] - x_avg) * (y_diff)
            y_var += y_diff * y_diff
            
    denom = math.sqrt(x_var * y_var)
    if denom == 0:
        return None
    return num / denom


def disparity_from_two_images(I1, I2, patch_border_size):
    assert patch_border_size % 2 == 1, "Both sides of the patch must be odd."
    
    b = int((patch_border_size - 1) / 2)
    
    r, c = I1.shape
    
    disparities = np.zeros(I1.shape)
        
    kernel_avg = np.ones((1, patch_border_size)) / patch_border_size
    kernel_sum = np.ones((1, patch_border_size))
        
    avg1 = cv2.filter2D(cv2.filter2D(I1, -1, kernel_avg), -1, kernel_avg.T)
    avg2 = cv2.filter2D(cv2.filter2D(I2, -1, kernel_avg), -1, kernel_avg.T)
    print(I1[b:b+10, b:b+10])
    print(avg1[b:b+10, b:b+10])
    
    
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(I1, cmap="gray")
    axes[1].imshow(I2, cmap="gray")
    plt.show()
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(avg1, cmap="gray")
    axes[1].imshow(avg2, cmap="gray")
    plt.show()

    diff1 = I1 - avg1
    diff2 = I2 - avg2
    
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(diff1, cmap="gray")
    axes[1].imshow(diff2, cmap="gray")
    plt.show()
   
    square1 = diff1 * diff1
    square2 = diff2 * diff2

    _, axes = plt.subplots(1, 2)
    axes[0].imshow(square1, cmap="gray")
    axes[1].imshow(square2, cmap="gray")
    plt.show()
    
    var1 = cv2.filter2D(cv2.filter2D(square1, -1, kernel_sum), -1, kernel_sum.T)
    var2 = cv2.filter2D(cv2.filter2D(square2, -1, kernel_sum), -1, kernel_sum.T)

    _, axes = plt.subplots(1, 2)
    axes[0].imshow(var1, cmap="gray")
    axes[1].imshow(var2, cmap="gray")
    plt.show()

    
    for y in range(b, r-b):
        belt = diff2[y-b : y+b+1]
        for x in range(b, c-b):
            num = cv2.filter2D(belt, -1, 
                               diff1[y-b:y+b+1, x-b:x+b+1],
                               borderType=cv2.BORDER_ISOLATED)
            num = num[b, b:c-b]
            denom = np.sqrt(var1[y, x] * var2[y, b:c-b])
            row = num / denom
            max_x = np.argmax(row)
            disparities[y, x] = x - max_x
        print(y)
    return disparities

"""     for y1 in range(b, r1-b):
        min_y = y1-b
        max_y = y1+b+1
        for x1 in range(b, c1-b):
            # we don't need y2, because we only search in the same line
            # in the first image, so one degree of freedom goes bye-bye
            reference_matrix = I1[min_y : max_y, 
                                  x1-b : x1+b+1]
            ref_avg = np.average(reference_matrix)
            ref_var = np.sum((reference_matrix - ref_avg)**2)


            max_NCC = -np.inf
            max_x = None
            for x2 in range(b, c2-b):
                second_matrix = I2[min_y : max_y, 
                                  x2-b : x2+b+1]
                NCC_value = NCC(reference_matrix, second_matrix, ref_avg, ref_var)
                if NCC_value != None and NCC_value > max_NCC:
                    max_NCC = NCC_value
                    max_x = x2
            if max_x == None:
                disparities[y1, x1] = 0
            else:
                disparities[y1, x1] = x1 - max_x          
            print(y1, x1)
    return disparities
 """
def case1d():
    dir = "data/disparity/"
    image_pair_names = ["cporta", "office", "office2"]
    for name in image_pair_names:
        name = dir + name + "_"
        image_left = imread(name + "left.png")
        image_right = imread(name + "right.png")
        disparities = disparity_from_two_images(
            to_gray(image_left), 
            to_gray(image_right),
            patch_border_size=21)
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(image_left)
        axes[0].set_title("Left image")
        axes[1].imshow(disparities, cmap="gray")
        axes[1].set_title("Disparities image")
        plt.show()


# 2

# 2b
def file_to_pts(arr2d):
    pts1 = []
    pts2 = []
    for x1, y1, x2, y2 in arr2d:
        pts1.append(np.array([x1, y1]))
        pts2.append(np.array([x2, y2]))
    return np.array(pts1), np.array(pts2)

def constructA(res1, res2):
    A = []
    for p1, p2 in zip(res1, res2):
        u = p1[0]
        v = p1[1]
        u_ = p2[0]
        v_ = p2[1]
        A.append(np.array([u * u_, u_ * v, u_, u * v_, v * v_,  v_, u, v, 1]))
    return np.array(A)

def fundamental_matrix(pts1, pts2):
    res1, T1 = normalize_points(pts1)
    res2, T2 = normalize_points(pts2)

    A = constructA(res1, res2)
    U, D, VT = np.linalg.svd(A)
    V = VT.T
    F = V[:,-1].reshape(3, 3)
    
    U, D, VT = np.linalg.svd(F)
    D[-1] = 0 # set the lowest eigenvalue to 0
    D = np.diag(D)
    
    F = np.matmul(np.matmul(U, D), VT)
    
    #U, _, VT = np.linalg.svd(F)
    #V = VT.T
    #e = V[:,-1] / V[-1, -1]
    #e_ = U[:,-1] / U[-1, -1]
    return np.matmul(np.matmul(T2.T, F), T1)


def draw_all_epilines(matrix, pts, h, w, axis):
    for pt in pts:
        x = pt[0]
        y = pt[1]
        homo_pt = np.array([x, y, 1])
        l = np.dot(matrix, homo_pt)
        draw_epiline(l, h, w, axis)
        
def pts_to_coord(pts):
    ptsx = []
    ptsy = []
    for pt in pts:
        ptsx.append(pt[0])
        ptsy.append(pt[1])
    return ptsx, ptsy

def case2b():
    dir = "data/epipolar/"
    pts = np.loadtxt(dir + "house_points.txt")
    pts1, pts2 = file_to_pts(pts)
    F = fundamental_matrix(pts1, pts2)
    print(F)
    
    house1 = imread(dir + "house1.jpg")
    house2 = imread(dir + "house2.jpg")
    h1, w1, *_ = house1.shape
    h2, w2, *_ = house2.shape
    
    pts1x, pts1y = pts_to_coord(pts1)
    pts2x, pts2y = pts_to_coord(pts2)

    _, axes = plt.subplots(1, 2)
    axes[0].imshow(house1)
    axes[0].scatter(pts1x, pts1y, color="r")
    draw_all_epilines(F.T, pts2, h1, w1, axes[0])
    axes[1].imshow(house2)
    axes[1].scatter(pts2x, pts2y, color="r")
    draw_all_epilines(F, pts1, h2, w2, axes[1])
    plt.show()
    

# 2c
def reprojection_error(F, pt1, pt2):
    pt1arr = np.array([pt1[0], pt1[1], 1])
    l = np.dot(F, pt1arr)
    a, b, c = l
    x0, y0 = pt2
    dist1 = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)

    pt2arr = np.array([pt2[0], pt2[1], 1])
    l = np.dot(F.T, pt2arr)
    a, b, c = l
    x0, y0 = pt1
    dist2 = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
    
    return (dist1 + dist2) / 2


def case2c():
    dir = "data/epipolar/"
    pts = np.loadtxt(dir + "house_points.txt")
    pts1, pts2 = file_to_pts(pts)
    F = fundamental_matrix(pts1, pts2)

    # (1)
    p1 = [85, 233]
    p2 = [67, 219]
    print("(1):")
    print(reprojection_error(F, p1, p2))
    print()
    
    print("(2):")
    sum = 0
    count = 0
    for pt1, pt2 in zip(pts1, pts2):
        sum += reprojection_error(F, pt1, pt2)
        count += 1
    print(sum / count)
    print()


# 2d
# this is hellinger distance between two histograms
def hellinger(h1, h2):
    sqrt1 = np.sqrt(h1)
    sqrt2 = np.sqrt(h2)
    diff_sqrt = sqrt1 - sqrt2
    square = diff_sqrt * diff_sqrt
    value_sumed = np.sum(square)
    value_half = value_sumed / 2
    return np.sqrt(value_half)

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

def align_feature_points_coordinates(arr, corres):
    aligned_arr = []
    for _, j in corres:
        aligned_arr.append(arr[j])
    return aligned_arr

def RANSAC(pts1, pts2, k=300, ransac_tsh=1):
    num_pts = len(pts1)
    
    # RANSAC loop
    best_F = None
    max_inliers = 0
    best_inliers  = []
    for _ in range(k):
        # 1. select 10 random correspondences
        pts1_copy = np.copy(pts1)
        pts2_copy = np.copy(pts2)
        pts1_selected = []
        pts2_selected = []
        for _ in range(10):
            idx = random.randint(0, pts1_copy.shape[0] - 1)
            pts1_selected.append(pts1[idx])
            pts2_selected.append(pts2[idx])
            pts1_copy = np.delete(pts1_copy, idx, 0)
            pts2_copy = np.delete(pts2_copy, idx, 0)
        pts1_selected = np.array(pts1_selected)
        pts2_selected = np.array(pts2_selected)
        
            
        # 2. estimate homography on those 10 random correspondences
        F = fundamental_matrix(pts1_selected, pts2_selected)
        
        # 3. project all other correspondences and check the number of inliers
        num_inliers = 0
        inliers = []
        for p1, p2 in zip(pts1, pts2):
            dist = reprojection_error(F, p1, p2)
            if dist < ransac_tsh:
                num_inliers += 1
                inliers.append((p1, p2))
                
                
        # 4. maximize the number of inliers and remember the correspondences
        # when the number of inliers was maximal
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F
            best_inliers = inliers

    print("Number of correspondences:", num_pts)
    print("Best F: ", best_F)
    print("Maximum number of inliers: ", max_inliers)
    
    return best_F, np.array(best_inliers)

def display_epilines_and_points(I1, pts1, I2, pts2, F):
    h1, w1, *_ = I1.shape
    h2, w2, *_ = I2.shape
    
    pts1x, pts1y = pts_to_coord(pts1)
    pts2x, pts2y = pts_to_coord(pts2)

    _, axes = plt.subplots(1, 2)
    axes[0].imshow(I1)
    axes[0].scatter(pts1x, pts1y, color="r")
    draw_all_epilines(F.T, pts2, h1, w1, axes[0])
    axes[1].imshow(I2)
    axes[1].scatter(pts2x, pts2y, color="r")
    draw_all_epilines(F, pts1, h2, w2, axes[1])
    plt.show()


def case2d():
    dir = "data/desk/"
    I1 = cv2.imread(dir + "DSC02638.jpg")
    I2 = cv2.imread(dir + "DSC02639.jpg")
    I1_gray= cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
    I2_gray= cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
    I1_color = cv2.cvtColor(I1,cv2.COLOR_BGR2RGB)
    I2_color = cv2.cvtColor(I2,cv2.COLOR_BGR2RGB)

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(I1_color)
    ax2.imshow(I2_color)
    plt.show()
    sift = cv2.SIFT_create()
    kp1, dsc1 = sift.detectAndCompute(I1_gray, None)
    kp2, dsc2 = sift.detectAndCompute(I2_gray, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(dsc1, dsc2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])
    print(good[0][0])
    pts1 = []
    pts2 = []
    for corr in good:
        corr = corr[0]
        pts1.append(np.array([int(kp1[corr.queryIdx].pt[0]), 
                int(kp1[corr.queryIdx].pt[1])]))
        pts2.append(np.array([int(kp2[corr.trainIdx].pt[0]), 
                int(kp2[corr.trainIdx].pt[1])]))
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    display_matches(I1_color, pts1, I2_color, pts2)
    
    F, inliers = RANSAC(pts1, pts2)
    
    num_points_drawn = 20
    pts1 = []
    pts2 = []
    i = 0
    for i in range(num_points_drawn):
        idx = random.randint(0, inliers.shape[0] - 1)
        pt1, pt2 = inliers[idx]
        pts1.append(pt1)        
        pts2.append(pt2)
        inliers = np.delete(inliers, idx, 0)
    
    display_epilines_and_points(I1_color, pts1, I2_color, pts2, F)
    
    
#case1b()
#case1d()

#case2b()
#case2c()
case2d()

#case3a()
#case3b()