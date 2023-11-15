import os
import a3_utils
import UZ_utils
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

def norm_odd(np_arr):
    return np_arr / np.sum(abs(np_arr))

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


#1

#1a
'''
    Solved in OneNote.
'''


#1b
def gauss_function_derivative(x, sigma):
    return - (1 / (math.sqrt(2 * math.pi) * (sigma ** 3))) * \
            x * math.exp(- (x ** 2)/(2 * (sigma ** 2)))

def gaussddx(sigma):
    upper = math.ceil(3 * sigma)
    whole = np.empty(2 * upper + 1)
    for x in range(-upper, upper + 1):
        whole[x + upper] = gauss_function_derivative(x, sigma)
    return norm_odd(whole), upper # normalization before return

def gauss_function(x, sigma):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * \
            math.exp(- (x ** 2)/(2 * (sigma ** 2)))

def gauss(sigma):
    upper = math.ceil(3 * sigma)
    second_half = np.empty(upper + 1)
    for x in range(upper + 1):
        second_half[x] = gauss_function(x, sigma)
    
    # in the kernel array we now only have the second half
    # of the array, so we need to shift the array and then
    # add the first half as the reversed second half without
    # the first element in the second half, because that is the
    # modus in the function and we don't want to duplicate it
    first_half = second_half[::-1][:-1]
    whole = np.concatenate([first_half, second_half])
    return norm(whole) # normalization before return

def case1b():
    gauss_derivative_kernel, upper = gaussddx(10)
    plt.plot(range(-upper, upper + 1), gauss_derivative_kernel)
    plt.show()


#1c
def convolute2D(image_gray, kernel1, kernel2 = None, 
                transposed1 = False, transposed2 = True):
    if kernel2 is None:
        kernel2 = np.copy(kernel1)
    # we turn both arrays, because cv2.filter2D performs
    # coorelation, not convolution:
    kernel1 = kernel1[::-1]
    kernel2 = kernel2[::-1]
    kernel1 = np.array([kernel1])
    kernel2 = np.array([kernel2])
    if transposed1:
        kernel1 = kernel1.T
    if transposed2:
        kernel2 = kernel2.T
    once_convoluted_image = cv2.filter2D(image_gray, -1, kernel1)
    return cv2.filter2D(once_convoluted_image, -1, kernel2)

def case1c():
    gauss_derivative_kernel, upper = gaussddx(5)
    gauss_kernel = gauss(5)
    #plt.plot(range(-upper, upper + 1), gauss_derivative_kernel)
    #plt.plot(range(-upper, upper + 1), gauss_kernel)
    #plt.show()

    *_, axes = plt.subplots(2, 3)

    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1
    axes[0, 0].imshow(impulse, cmap="gray")
    axes[0, 0].set_title('impulse')

    #(a)
    '''
        You get a narrower 2D gaussian kernel.
    '''
    impulseA = convolute2D(impulse, gauss_kernel)
    axes[1, 0].imshow(impulseA, cmap="gray")
    axes[1, 0].set_title('G, Gt')

    #(b)
    '''
        Derivate of impulse image by y.
    '''
    impulseB = convolute2D(impulse, gauss_kernel, 
                           gauss_derivative_kernel)
    axes[0, 1].imshow(impulseB, cmap="gray")
    axes[0, 1].set_title('G, Dt')
   
    #(c)
    '''
        Derivate of impulse image by x.
    '''
    impulseC = convolute2D(impulse, gauss_derivative_kernel, 
                           gauss_kernel)
    axes[1, 1].imshow(impulseC, cmap="gray")
    axes[1, 1].set_title('Gt, D')

    #(d)
    '''
        Derivate of impulse image by x.
    '''
    impulseD = convolute2D(impulse, gauss_kernel, 
                           gauss_derivative_kernel,
                           transposed1=True, transposed2=False)
    axes[0, 2].imshow(impulseD, cmap="gray")
    axes[0, 2].set_title('D, Gt')

    #(e)
    '''
        Derivate of impulse image by y.
    '''
    impulseE = convolute2D(impulse, gauss_derivative_kernel,
                           gauss_kernel, 
                           transposed1=True, transposed2=False)
    axes[1, 2].imshow(impulseE, cmap="gray")
    axes[1, 2].set_title('Dt, G')
    
    plt.show()

    # the order of operations is not important, because convolution
    # is comutative (G, Dt has the same solution as Dt, G and that
    # also holds for Gt, D and D, Gt)


#1d
def first_derivatives_of_image(image_gray):
    gauss_kernel = gauss(5)
    gauss_derivative_kernel = gaussddx(5)[0]
    
    der_x = convolute2D(image_gray, gauss_derivative_kernel, 
                           gauss_kernel)
    der_y = convolute2D(image_gray, gauss_kernel, 
                           gauss_derivative_kernel)
    
    return der_x, der_y

def second_derivatives_of_image(image_gray):
    der_x, der_y = first_derivatives_of_image(image_gray)
    der_xx, der_xy = first_derivatives_of_image(der_x)
    _, der_yy = first_derivatives_of_image(der_y)
    return der_xx, der_xy, der_yy

def gradient_magnitude(image_gray):
    der_x, der_y = first_derivatives_of_image(image_gray)
    magnitudes = np.sqrt((der_x * der_x) + (der_y * der_y))
    angles = np.arctan2(der_y, der_x)
    return magnitudes, angles

def case1d():
    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1

    museum = imread('images/museum.jpg')
    museum_gray = to_gray(museum)
    
    der_x, der_y = first_derivatives_of_image(museum_gray)
    der_xx, der_xy, der_yy = second_derivatives_of_image(museum_gray)
    mags, angles = gradient_magnitude(museum_gray)
    
    _, axes = plt.subplots(2, 4)
    images = np.array([[museum_gray, der_x, der_y, mags], 
                       [der_xx, der_xy, der_yy, angles]])
    titles = np.array([['Original', 'I_x', 'I_y', 'I_mag'], 
                       ['I_xx', 'I_xy', 'I_yy', 'I_dir']])
    for i in range(2):
        for j in range(4):
            axes[i, j].imshow(images[i, j], cmap="gray")
            axes[i, j].set_title(titles[i, j])
    plt.show()


#1e
def gridHist(image_grid):
    mags, angles = gradient_magnitude(image_grid)
    num_bins = 8
    max_angle = np.pi
    min_angle = - np.pi
    range_angle = max_angle - min_angle # pi - (-pi) = 2pi
    bin_size = range_angle / num_bins # 2pi / 8 = pi/4
    H = np.zeros(num_bins)
    round_down_vectorized = np.vectorize(round_down)
    indexes = round_down_vectorized((angles - min_angle) / bin_size)
    indexes -= (indexes == num_bins)
    
    for i in range(num_bins):
        sel = indexes == i
        mags_to_sum = mags * sel
        H[i] = np.sum(mags_to_sum)
    
    return H
    


def myhist3Dgradient(image_colored):
    image_gray = to_gray(image_colored)
    r, c, *_ = image_gray.shape
    grid_r = int(r / 8)
    grid_c = int(c / 8)
    hist = np.array([])
    for i in range(8):
        for j in range(8):
            hist_grid = gridHist(image_gray[grid_r*i:grid_r*(i+1),
                                            grid_c*j:grid_c*(j+1)])
            hist = np.concatenate([hist, hist_grid])
    return norm(hist)

def myhist3D(image_colored, num_bins, mymin = 0, mymax = 1):
    myrange = mymax - mymin
    bin_size = myrange / num_bins
    H = np.zeros((num_bins, num_bins, num_bins))
    
    round_down_vectorized = np.vectorize(round_down)
    indexes = round_down_vectorized((image_colored - mymin) / bin_size)
    indexes -= (indexes == num_bins)
    flattened_indexes = indexes.reshape((-1, 3))
    
    unique, counts = np.unique(flattened_indexes, axis=0, return_counts=True)
    H[unique[:, 0], unique[:, 1], unique[:, 2]] = counts
    
    return norm(H)

def myhist1D(image_colored, num_bins, gradient):
    if gradient:
        return myhist3Dgradient(image_colored)
    else:
        return myhist3D(image_colored, num_bins, 0, 1).reshape(-1)

def get_histograms1D(folder_path, num_bins, gradient):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter only files with image extensions
    image_files = [f for f in files if f.lower().
                   endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    images_and_histograms = []
    
    # Open and process each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        image = imread(image_path)
        hist = myhist1D(image, num_bins, gradient)
        images_and_histograms.append((image_file, image, hist))
        #print(images_and_histograms)
    return images_and_histograms

dist_measures = ["l2", "chi", "inter", "hell"]

def euclidean(h1, h2):
    diff = h1 - h2
    square = diff * diff
    value_sumed = np.sum(square)
    return np.sqrt(value_sumed)

def chi_squared(h1, h2):
    diff = h1 - h2
    square = diff * diff
    numerator = square
    
    sumed = h1 + h2
    epsilon = 1e-10
    denominator = sumed + epsilon

    fraction = numerator / denominator
    value_sumed = np.sum(fraction)
    return value_sumed / 2

def intersection(h1, h2):
    diff = h1 - h2
    mask1 = diff < 0
    mask2 = ~mask1
    minimum = h1 * mask1 + h2 * mask2
    value_sumed = np.sum(minimum)
    return 1 - value_sumed

def hellinger(h1, h2):
    sqrt1 = np.sqrt(h1)
    sqrt2 = np.sqrt(h2)
    diff_sqrt = sqrt1 - sqrt2
    square = diff_sqrt * diff_sqrt
    value_sumed = np.sum(square)
    value_half = value_sumed / 2
    return np.sqrt(value_half)

def compare_histograms(h1, h2, str):
    if (str == "l2"):
        return euclidean(h1, h2)
    elif (str == "chi"):
        return chi_squared(h1, h2)
    elif (str == "inter"):
        return intersection(h1, h2)
    elif (str == "hell"):
        return hellinger(h1, h2)
    else:
        print(textwrap.dedent("""\
            Not allowed string for distance measure, try again.\n
            Here are the string commands that are allowed:\n
                l2 -> Euclidean distance\n
                chi -> Chi_square distance\n
                inter -> Intersection\n
                hell -> Hellinger distance"""))
        return -1

def sorting_atribute(e):
    return e[3]

def assignment2case3d(num_bins_per_dim = 8, gradient=False):
    images_and_histograms = get_histograms1D("../assignment2/dataset", 
                                             num_bins_per_dim, gradient)
    reference_image_and_histogram = images_and_histograms[19]
    reference_histogram = reference_image_and_histogram[2]
    
    distance_measure = dist_measures[3]

    
    
    i = 0
    for image_and_hist in images_and_histograms:
        hist = image_and_hist[2]
        dist = compare_histograms(reference_histogram,
                                  hist,
                                  distance_measure)
        images_and_histograms[i] = (image_and_hist[0],
                                    image_and_hist[1],
                                    image_and_hist[2],
                                    dist)
        i += 1
    
    #print(images_and_histograms)
    images_and_histograms.sort(key=sorting_atribute)
    
    closest_images_and_histograms = images_and_histograms[:6]
    ciah = closest_images_and_histograms
    
    *_, axes = plt.subplots(2, 6)
            
    for i in range(6):
        axes[0, i].imshow(ciah[i][1])
        axes[0, i].set_title(ciah[i][0])
        axes[1, i].bar(range(ciah[i][2].shape[0]), ciah[i][2], width=5)
        axes[1, i].set_title(f"{distance_measure}={ciah[i][3]:.2f}")
    plt.show()

def case1e():
    assignment2case3d(gradient=True)
    
    
    
    
    
        
# this is where you run cases:

#case1b()
#case1c()
#case1d()
case1e()

#case2a()
#case2b()
#case2c()

#case3a()
#case3b()
#case3c()
#case3d()
#case3e()
#case3f()
#case3g()
#case3h()