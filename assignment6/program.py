from a6_utils import *
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2



# 1

# 1a
def direct_PCA(vectors):
    N = vectors.shape[0]
    matrix = vectors
    mu = np.mean(matrix, axis=0)
    matrix_centered = matrix - mu
    matrix_centered_T = matrix_centered.T
    covariance_matrix = 1 / (N - 1) * np.dot(matrix_centered_T, matrix_centered_T.T)
    U, S, _ = np.linalg.svd(covariance_matrix)
    return U.T, mu, covariance_matrix, S
    
def case1a():
    A = [3, 4]
    B = [3, 6]
    C = [7, 6]
    D = [6, 4]
    points = np.asarray([A, B, C, D])
    eigenvectors, center, *_ = direct_PCA(points)
    
    plt.scatter(0, 0, alpha=0)
    plt.scatter(points[:, 0], points[:, 1], c="black")
    for eigenvector in eigenvectors:
        plt.plot([center[0], center[0] + eigenvector[0]], [center[1], center[1] + eigenvector[1]])
    plt.show()


# 1b
def visualize(points, eigenvectors, center, C, S, show_eigenvectors=False):
    bottom = np.min(points)
    upper = np.max(points)
    plt.xlim(bottom - 1, upper + 1)
    plt.ylim(bottom - 1, upper + 1)
    plt.scatter(0, 0, alpha=0)
    plt.scatter(points[:, 0], points[:, 1], c="black")
    if show_eigenvectors:
        S_expanded = np.expand_dims(S, axis=1)
        #print(eigenvectors)
        #print(S_expanded)
        scaled_eigenvectors = eigenvectors * S_expanded
        #print(scaled_eigenvectors)
        
        colors = ["red", "green"]
        for i, eigenvector in enumerate(scaled_eigenvectors):
            plt.plot([center[0], center[0] + eigenvector[0]], 
                     [center[1], center[1] + eigenvector[1]], 
                     color=colors[i])
    drawEllipse(center, C)
    plt.show()

def calc_and_visualize(points, show_eigenvectors=False):
    eigenvectors, center, C, S = direct_PCA(points)
    visualize(points, eigenvectors, center, C, S, show_eigenvectors)
    
def case1b():
    points = np.loadtxt("data/points.txt")
    print(points)
    calc_and_visualize(points)


# 1c
def case1c():
    points = np.loadtxt("data/points.txt")
    calc_and_visualize(points, show_eigenvectors=True)    


# 1d
def normalize_to_max(arr):
    return arr / np.max(arr)

def normalize_to_sum(arr):
    return arr / np.sum(arr)

def case1d():
    points = np.loadtxt("data/points.txt")
    print(points)
    *_, S = direct_PCA(points)
    plt.bar(x=range(len(S)), height=normalize_to_max(S))
    plt.show()
    
    S_norm = normalize_to_sum(S)
    print("Amount of information we retain:", 
          np.round(np.sum(S_norm[:-1]) * 100, 2), 
          "%")


# 1e
def case1e():
    points = np.loadtxt("data/points.txt")
    print(points)
    UT, mu, C, S = direct_PCA(points)
    U = UT
    U[:, 1] = 0
    
    points_subspace = [np.dot(U.T, point - mu) for point in points]
    points_subspace_back = np.asarray([np.dot(U, point) + mu for point in points_subspace])

    visualize(points_subspace_back, U.T, mu, C, S, show_eigenvectors=True)


# 1f
def euclidean_dist(arr1, arr2):
    arr_diff = arr1 - arr2
    arr_sq = arr_diff * arr_diff
    return np.sqrt(np.sum(arr_sq))

def closest_pt_print(points, q, print_str, before):
    min = np.inf
    closest_pt = None
    idx = None
    for i, pt in enumerate(points):
        dist = euclidean_dist(pt, q)
        if dist < min:
            min = dist
            closest_pt = pt
            idx = i
    if before:
        points_names = ['A', 'B', 'C', 'D', 'E']
    else:
        points_names = ['A\'', 'B\'', 'C\'', 'D\'', 'E\'']
        
    print(print_str, points_names[idx], closest_pt, "with distance", np.round(min, 2))

def visualize_reconstruction(points_before, center, C,
                             points_after, q_before=None, q_after=None, 
                             is_q=False, is_elipse=True):
    bottom = np.min(points_before)
    upper = np.max(points_before)
    #plt.xlim(bottom - 1, upper + 1)
    #plt.ylim(bottom - 1, upper + 1)
    plt.scatter(0, 0, alpha=0)
    plt.scatter(points_before[:, 0], points_before[:, 1], c="blue", s=5)
    plt.scatter(points_after[:, 0], points_after[:, 1], c="green", s=5)
    args={"color":"red", "width": 0.01, "length_includes_head":True}
    for i, (pt_before, pt_after) in enumerate(zip(points_before, points_after)):
        dist = euclidean_dist(pt_after, pt_before)
        hl = dist * 0.2
        hw = dist * 0.04
        if dist < hl:
            hl = dist
            hw = 0.001
        
        plt.arrow(pt_before[0], pt_before[1], 
                  pt_after[0] - pt_before[0], pt_after[1] - pt_before[1], 
                  **args, head_length=hl, head_width=hw)
        plt.annotate(chr(i + 65), pt_before)
        plt.annotate(chr(i + 65)+'\'', pt_after)
          
    if is_q:
        plt.scatter(q_before[0], q_before[1], c="blue")
        plt.scatter(q_after[0], q_after[1], c="green")
        
        hl = 0.5
        hw = 0.1
        dist = euclidean_dist(q_after, q_before)
        if dist < hl:
            hl = dist
            hw = 0.001
        plt.arrow(q_before[0], q_before[1], 
                q_after[0] - q_before[0], q_after[1] - q_before[1], 
                **args, head_length=hl, head_width=hw)
        plt.annotate('q', q_before)
        plt.annotate('q\'', q_after)
    if is_elipse:
        drawEllipse(center, C)
    
    #draw a line through the points, 
    # so that it is more easily seen that they are all colinear
    first_point = np.min(points_after, axis=0)
    last_point = np.max(points_after, axis=0)
    plt.plot([first_point[0], last_point[0]], [first_point[1], last_point[1]], 
             color="purple")
    
    plt.show()
    
def case1f():
    points = np.loadtxt("data/points.txt")
    print(points)
    q = np.asarray([6, 6])
    print(np.round(euclidean_dist([0, 0], q), 2))
    closest_pt_print(points, q, "Closest point before PCA:", before=True)
    
    UT, mu, C, S = direct_PCA(points)
    U = UT
    print(U)
    visualize(points, U.T, mu, C, S)
    
    U_tilde = np.copy(U)
    U_tilde[:, 1] = 0
    points_subspace = [np.dot(U_tilde.T, point - mu) for point in points]
    points_subspace_back = np.asarray([np.dot(U_tilde, point) + mu for point in points_subspace])
    q_subspace = np.dot(U_tilde.T, q - mu)
    q_subspace_back = np.dot(U_tilde, q_subspace) + mu
    closest_pt_print(points_subspace_back, q_subspace_back, 
                     "Closest point after PCA is", before=False)
    visualize(points_subspace_back, U_tilde.T, mu, C, S)
    
    visualize_reconstruction(points, mu, C, points_subspace_back, 
                             q_before=q, q_after=q_subspace_back, is_q=True)


# 2

# 2a
def dual_PCA(points, change=False, idx1=None, idx2=None):
    N = points.shape[0]
    matrix = points
    mu = np.mean(matrix, axis=0)
    matrix_centered = matrix - mu
    matrix_centered_T = matrix_centered.T
    print("matrix_centered_T.shape:", matrix_centered_T.shape)
    covariance_matrix = 1 / (N - 1) * np.dot(matrix_centered_T.T, matrix_centered_T)
    print("C\'.shape:", covariance_matrix.shape)
    U, S, _ = np.linalg.svd(covariance_matrix)
    #print()
    #print(U)
    #print()
    #print(S)
    print("U\'.shape:", U.shape)
    print("S\'.shape:", S.shape)
    for i in range(S.shape[0]):
        if S[i] == 0:
            S[i] = 1e-15
    S_inv = np.linalg.inv(np.diag(S))
    rhs = np.sqrt(1 / (N - 1) * S_inv)
    #print()
    #print(rhs)
    lhs = np.dot(matrix_centered_T, U)
    #print()
    #print(lhs)
    U = np.dot(lhs, rhs)
    #print()
    #print(U)
    print("U.shape:", U.shape)
    
    points_subspace = np.asarray([np.dot(U.T, point - mu) for point in points])
    print()
    print("points_subspace.shape:", points_subspace.shape)
        
    if change:
        print(points_subspace.shape)
        points_subspace[idx1][idx2] = 0
    points_subspace_back = np.asarray([np.dot(U, point) + mu for point in points_subspace])
    print("points_subspace_back.shape:", points_subspace_back.shape)
    return U.T, mu, covariance_matrix, S, points_subspace_back

def case2a():
    points = np.loadtxt("data/points.txt")
    print(points)
    _, mu, C, _, points_after = dual_PCA(points)
    print(points_after)
    visualize_reconstruction(points, mu, C, points_after, is_elipse=False)


# 2b
def case2b():
    points = np.loadtxt("../assignment5/data/epipolar/house_points.txt")
    points_left = points[:, :2]
    points_right = points[:, 2:]
    print()
    print(points_left)
    print()
    print(points_right)
    
    _, mu, C, _, points_left_after = dual_PCA(points_left)
    print(points_left_after)
    visualize_reconstruction(points_left, mu, C, points_left_after, is_elipse=False)

    _, mu, C, _, points_right_after = dual_PCA(points_right)
    print(points_right_after)
    visualize_reconstruction(points_left, mu, C, points_right_after, is_elipse=False)
    
    
# 3

# 3a
def case3a():
    folder_path = "data/faces/1"
    file_list = os.listdir(folder_path)
    image_files = [file for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # reshape into a column
        shapes = image.shape
        image = np.reshape(image, (-1, 1))
        if i == 0:
            images = image
        else:
            images = np.hstack((images, image))

    print(images)
    print(images.shape)
    return images, shapes


def case3b():
    images, shapes = case3a()
    UT, mu, C, S, images_after = dual_PCA(images.T)
    images_after = images_after.T
    print(images_after)
    print(images_after.shape)
    for i in range(5):
        image = np.reshape(UT[i], shapes)
        plt.subplot(151 + i)
        plt.imshow(image, cmap="gray")
    plt.show()
    
    image1 = images[:, 0]
    image2 = np.copy(image1)
    image3 = np.copy(image2)
    
    print("DAMN")
    print(image1.shape)
    
    image2[4074] = 0
    
    images123 = np.vstack((image1, image2, image3))
    print(images123)
    print(images123.shape)
    *_, images123_after = dual_PCA(images123, change=True, idx1=2, idx2=1)

    for i in range(len(images123_after)):
        image = images123_after[i]
        image = np.reshape(image, shapes)
        plt.subplot(101 + 10 * len(images123_after) + i)
        plt.imshow(image, cmap="gray")
    plt.show()
    
    """
    How many pixels are different in the second and the third image
    relative to the first image?
    """
    image1_after = images123_after[0]
    image2_after = images123_after[1]
    image3_after = images123_after[2]
    print("Diff in pixels in the first and second image:", np.sum(1 - (image1_after == image2_after)))
    print("Diff in pixels in the first and third image:", np.sum(1 - (image1_after == image3_after)))


#def case3c():

#case1a()
#case1b()
#case1c()
#case1d()
#case1e()
#case1f()

#case2a()
#case2b()

#case3a()
case3b()
#case3c()