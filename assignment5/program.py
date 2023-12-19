from a5_utils import *
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


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
    if num == 0 or denom == 0:
        return None
    return num / denom

def disparity_from_two_images(I1, I2, patch_border_size):
    assert patch_border_size % 2 == 1, "Both sides of the patch must be odd."
    
    b = int((patch_border_size - 1) / 2)
    
    h1, c1, *_ = I1.shape
    _, c2, *_ = I2.shape
    
    disparities = np.empty(I1.shape)
    
    disparities[:b, :] = 0 # top horizontal black edge
    disparities[h1-b:, :] = 0 # bottom horizontal black edge
    
    disparities[:, :b] = 0 # left vertical black edge
    disparities[:, c1-b:] = 0 # right vertical black edge

    for y1 in range(b, h1-b):
        for x1 in range(b, c1-b):
            # we don't need y2, because we only search in the same line
            # in the first image, so one degree of freedom goes bye-bye
            reference_matrix = I1[y1-b : y1+b+1, 
                                  x1-b : x1+b+1]
            ref_avg = np.average(reference_matrix)
            ref_var = np.sum((reference_matrix - ref_avg)**2)


            max_NCC = -np.inf
            max_x = None
            y2 = y1
            for x2 in range(b, c2-b):
                second_matrix = I2[y2-b : y2+b+1, 
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
            patch_border_size=11)
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(image_left)
        axes[0].set_title("Left image")
        axes[1].imshow(disparities)
        axes[1].set_title("Disparities image")
        plt.show()


#case1b()
case1d()
#case1e()

#case2a()
#case2b()
#case2c()
#case2d()

#case3a()
#case3b()