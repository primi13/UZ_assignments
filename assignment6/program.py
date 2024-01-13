from a6_utils import *
import numpy as np
import matplotlib.pyplot as plt



# 1

# 1a
def direct_PCA(vectors):
    N = vectors.shape[0]
    matrix = vectors
    mu = np.mean(matrix, axis=0)
    matrix_centered = matrix - mu
    matrix_centered_T = matrix_centered.T
    covariance_matrix = 1 / (N - 1) * np.dot(matrix_centered_T, matrix_centered_T.T)
    U, _, _ = np.linalg.svd(covariance_matrix)
    return U.T, mu
    
def case1a():
    A = [3, 4]
    B = [3, 6]
    C = [7, 6]
    D = [6, 4]
    points = np.asarray([A, B, C, D])
    eigenvectors, center = direct_PCA(points)
    
    plt.scatter(0, 0, alpha=0)
    plt.scatter(points[:, 0], points[:, 1], c="black")
    for eigenvector in eigenvectors:
        plt.plot([center[0], eigenvector[0] + center[0]], [center[1], eigenvector[1] + center[1]])
    plt.show()


# 1b
def case1b():
    points = np.loadtxt("data/points.txt")
    print(points)

    eigenvectors, center = direct_PCA(points)
    
    plt.scatter(0, 0, alpha=0)
    plt.scatter(points[:, 0], points[:, 1], c="black")
    for eigenvector in eigenvectors:
        plt.plot([center[0], eigenvector[0] + center[0]], [center[1], eigenvector[1] + center[1]])
    plt.show()


# 1c
def case1c():
    



#case1a()
#case1b()
#case1c()