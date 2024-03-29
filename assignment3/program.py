import os
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# function from a3_utils, because I changed it
def draw_line(rho, theta, h, w, axis = None, linewidth=0.7):
    """
    Example usage:

    plt.imshow(I)
    draw_line(rho1, theta1, h, w)
    draw_line(rho2, theta2, h, w)
    draw_line(rho3, theta3, h, w)
    plt.show()

    "rho" and "theta": Parameters for the line which will be drawn.
    "h", "w": Height and width of an image.
    """

    c = np.cos(theta)
    s = np.sin(theta)

    xs = []
    ys = []
    if s != 0:
        y = int(rho / s)
        if 0 <= y < h:
            xs.append(0)
            ys.append(y)

        y = int((rho - w * c) / s)
        if 0 <= y < h:
            xs.append(w - 1)
            ys.append(y)
    if c != 0:
        x = int(rho / c)
        if 0 <= x < w:
            xs.append(x)
            ys.append(0)

        x = int((rho - h * s) / c)
        if 0 <= x < w:
            xs.append(x)
            ys.append(h - 1)

    if axis is not None:
        axis.plot(xs[:2], ys[:2], 'r', linewidth=linewidth)
    else:    
        plt.plot(xs[:2], ys[:2], 'r', linewidth=linewidth)


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
    gauss_derivative_kernel, upper = gaussddx(4)
    gauss_kernel = gauss(4)
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
def first_derivatives_of_image(image_gray, sigma=1):
    gauss_kernel = gauss(sigma)
    gauss_derivative_kernel = gaussddx(sigma)[0]
    
    der_x = convolute2D(image_gray, gauss_derivative_kernel, 
                           gauss_kernel)
    der_y = convolute2D(image_gray, gauss_kernel, 
                           gauss_derivative_kernel)
    
    return der_x, der_y

def second_derivatives_of_image(image_gray, sigma=1):
    der_x, der_y = first_derivatives_of_image(image_gray, sigma)
    der_xx, der_xy = first_derivatives_of_image(der_x, sigma)
    _, der_yy = first_derivatives_of_image(der_y, sigma)
    return der_xx, der_xy, der_yy

def gradient_magnitude(image_gray, sigma):
    der_x, der_y = first_derivatives_of_image(image_gray, sigma)
    magnitudes = np.sqrt((der_x * der_x) + (der_y * der_y))
    angles = np.arctan2(der_y, der_x)
    return magnitudes, angles

def case1d():
    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1

    museum = imread('images/museum.jpg')
    museum_gray = to_gray(museum)
    
    der_x, der_y = first_derivatives_of_image(museum_gray, sigma=1)
    der_xx, der_xy, der_yy = second_derivatives_of_image(museum_gray, 
                                                         sigma=1)
    mags, angles = gradient_magnitude(museum_gray, 1)
    
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
    mags, angles = gradient_magnitude(image_grid, 1)
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
    


#2

#2a
def normal_thresholding(image_gray, threshold):
    image_mask = np.copy(image_gray)
    image_mask[image_gray < threshold] = 0
    return image_mask

def normalizeValues(arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    return (arr - min_val) / (max_val - min_val)

def findedges(image_gray, sigma, theta, normalize=False):
    mags, angles = gradient_magnitude(image_gray, sigma)
    if normalize:
        mags = normalizeValues(mags)
    return normal_thresholding(mags, theta), mags, angles

def case2a():
    museum_gray = to_gray(imread('images/museum.jpg'))
    museum_masks = [] # it will be list of length 8
    for i in np.arange(0.0, 0.25, 0.025):
        print(i)
        museum_masks.append(findedges(museum_gray, 2, i)[0])
    _, axes = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            if i == 1 and j == 4:
                axes[i, j].imshow(museum_gray, cmap="gray",
                                vmin=0, vmax=1)
                axes[i, j].set_title("Gray museum")
            else:
                axes[i, j].imshow(museum_masks[i*5 + j], cmap="gray",
                                vmin=0, vmax=1)
                axes[i, j].set_title(f"tsh = {(i * 5 + j) / 40}")
    plt.show()
    
             
#2b
def angle_to_neighbors(angle):
    increments = [(0, -1), (-1, -1), (-1, 0), (1, -1), 
                  (0, 1), (1, 1), (1, 0), (-1, 1)]
    angle = (angle + (9*np.pi)/8) % (2 * np.pi)
    index = (np.floor(angle / (np.pi / 4))).astype(int)
    return increments[index]

def non_maxima_suppresion(mags, angles):
    image_thinned = np.copy(mags)
    r, c, *_ = image_thinned.shape
    for i in range(r):
        for j in range(c):
            angle = angles[i, j]            
            i_inc, j_inc = angle_to_neighbors(angle)
            i_new_1 = i + i_inc
            j_new_1 = j + j_inc
            if i_new_1 > 0 and i_new_1 < r and j_new_1 > 0 and j_new_1 < c:
                if mags[i_new_1, j_new_1] > mags[i, j]:
                    image_thinned[i, j] = 0
                    continue
            
            i_new_2 = i - i_inc
            j_new_2 = j - j_inc
            if i_new_2 > 0 and i_new_2 < r and j_new_2 > 0 and j_new_2 < c:
                if mags[i_new_2, j_new_2] > mags[i, j]:
                    image_thinned[i, j] = 0
    return image_thinned
            

def case2b():
    museum_gray = to_gray(imread('images/museum.jpg'))
    museum_mask, mags, angles = findedges(museum_gray, 1, 0.16)
    museum_thinned = non_maxima_suppresion(mags, angles)
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(museum_mask, cmap="gray")
    axes[1].imshow(museum_thinned, cmap="gray")
    plt.show()
    
    
#2c
def hystresis_thresholding(num_labels, label_array, image_gray, tsh_low, tsh_high):
    new_image_mask = np.copy(image_gray)
    for i in range(1, num_labels):
        mask = label_array == i
        label_cut = new_image_mask[mask]
        if np.max(label_cut) < tsh_high:
            label_cut = 0
        else:
            low_mask = label_cut < tsh_low
            label_cut[low_mask] = 0
            label_cut[~low_mask] = 1
        new_image_mask[mask] = label_cut
    return new_image_mask.astype(np.uint8)

def case2c():
    museum_gray = to_gray(imread('images/museum.jpg'))
    tsh = 0.04
    mags_tsh, _, angles = findedges(museum_gray, sigma=1, theta=tsh,
                                    normalize=True)
    museum_thinned = non_maxima_suppresion(mags_tsh, angles)
    output = cv2.connectedComponentsWithStats(
        (museum_thinned * 255).astype(np.uint8)) #connectivity=8
    tsh_low = 0.15
    tsh_high = 0.30
    museum_hysteresised = hystresis_thresholding(output[0], 
                                                 output[1], 
                                                 museum_thinned,
                                                 tsh_low=tsh_low,
                                                 tsh_high=tsh_high)
    _, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(museum_gray, cmap="gray")
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(mags_tsh, cmap="gray")
    axes[0, 1].set_title(f'Thresholded (thr = {tsh})')
    axes[1, 0].imshow(museum_thinned, cmap="gray")
    axes[1, 0].set_title(f'Nonmax. supp. (thr = {tsh})')
    axes[1, 1].imshow(museum_hysteresised, cmap="gray")
    axes[1, 1].set_title(f"Hysteresis (high = {tsh_high}, " + 
                         f"low = {tsh_low})")
    plt.show()


#3

#3a
def lines_through_point(x: int, y: int, acc, D):
    # map theta = -pi/2 to -num_bins/2 and pi/2 to num_bins_2 linearly
    r, c, *_ = acc.shape
    '''this will be an array of zeroes and ones and this array will then
    be added to the input array'''
    local_array = np.zeros((r, c), dtype=np.int8)
    theta_increment = np.pi / c
    i = 0 # represents theta coordinate (x coordinate), which is a whole number
    for theta in np.arange(-np.pi/2, np.pi/2, theta_increment):
        rho = x * np.cos(theta) + y * np.sin(theta)
        # normalizing rho to interval [0, r], so that the lines fall inside
        # of the graph
        rho_on_graph = int((rho - (-D)) / (D - (-D)) * r)
        local_array[rho_on_graph, i] = 1
        i += 1
                
    acc += local_array


def case3a():
    #create an accumulator array
    num_bins = 300
    coord = np.array([[(10, 10), (30, 60)], [(50, 20), (80, 90)]])
    acc = np.zeros((num_bins, num_bins))
    accs = np.array([[acc.copy(), acc.copy()], [acc.copy(), acc.copy()]])
    _, axes = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            x, y = coord[i, j]
            lines_through_point(x, y, accs[i, j], 150)
            axes[i, j].imshow(accs[i, j])
            axes[i, j].set_title(f"x = {x}, y = {y}")
    plt.show()
    
    

#3b
def normal_thresholding2(arr, tsh):
    arr_new = arr.copy()
    arr_new[arr_new < tsh] = 0
    arr_new[arr_new >= tsh] = 1
    return arr_new

def hough_find_lines(num_bins_theta, num_bins_rho, image_gray, 
                     sigma, tsh, use_angles=False, radius=None):
    image_mags_mask, _, angles = findedges(image_gray, sigma, 
                                           tsh, normalize=True)
    acc = np.zeros((num_bins_rho, num_bins_theta))
    r, c, *_ = image_mags_mask.shape
    D = np.sqrt(r * r + c * c)
    
    if not use_angles and radius is None:
        for y in range(r):
            for x in range(c):
                if image_mags_mask[y, x] > 0:
                    lines_through_point(x, y, acc, D)
    elif radius is None:
        for y in range(r):
            for x in range(c):
                if image_mags_mask[y, x] > 0:
                    lines_through_point_fixed_theta(x, y, acc, 
                                                   D, angles[y, x])
    else:
        for y in range(r):
            for x in range(c):
                if image_mags_mask[y, x] > 0:
                    circles_through_point(x, y, acc, D, radius)
    return normalizeValues(acc)


def case3b():
    num_bins_theta = 200
    num_bins_rho = 200
    tsh_find_edges = 0.7
    
    synthetic = np.zeros((100, 100))
    synthetic[10, 10] = 1
    synthetic[10, 20] = 1
    acc = hough_find_lines(num_bins_theta, num_bins_rho, 
                           synthetic, 1, tsh_find_edges)
    _, axes = plt.subplots(1, 3)
    axes[0].imshow(acc)
    axes[0].set_title("Synthetic")
    
    oneline = to_gray(imread("images/oneline.png"))
    acc = hough_find_lines(num_bins_theta, num_bins_rho, 
                           oneline, 1, tsh_find_edges)
    axes[1].imshow(acc)
    axes[1].set_title("Oneline")

    rectangle = to_gray(imread("images/rectangle.png"))
    acc = hough_find_lines(num_bins_theta, num_bins_rho, 
                           rectangle, 1, tsh_find_edges)
    axes[2].imshow(acc)
    axes[2].set_title("Rectangle")
    plt.show()


#3c
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

def case3c():
    num_bins_theta = 200
    num_bins_rho = 200
    tsh_find_edges = 0.7
    oneline = to_gray(imread("images/oneline.png"))
    acc = hough_find_lines(num_bins_theta, num_bins_rho, 
                           oneline, 1, tsh_find_edges)
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(acc)
    axes[0].set_title("Oneline")
    acc = nonmaxima_suppression_box(acc)
    axes[1].imshow(acc)
    axes[1].set_title("Oneline - only local maxima")
    plt.show()

    
#3d
def get_rho_back(rho_on_graph, acc_r, r, c):
    D = np.sqrt(r * r + c * c)
    return ((rho_on_graph * 2 * D) / acc_r) - D

def get_theta_back(i, acc_c):
    return -np.pi/2 + np.pi / acc_c * i

def sort_acc_atr(e):
    return e[0]

def visualize_the_acc(arr, r, c, axis):
    l = len(arr)
    acc = np.zeros((r, c))
    for i in range(l):
        acc[arr[i][1], arr[i][2]] = 1
    axis.imshow(acc)

def draw_lines_for_image(image_gray, axis, image_name=None, sigma=1, 
                         num_bins_theta=200, num_bins_rho=200, 
                         tsh_find_edges=0.7, tsh_final=1, n=None,
                         use_angles=False):

    acc = hough_find_lines(num_bins_theta, num_bins_rho, 
                           image_gray, sigma, tsh_find_edges,
                           use_angles)
        
    acc_maxima = nonmaxima_suppression_box(acc)
            
    acc_r, acc_c, *_ = acc_maxima.shape
    r, c, *_ = image_gray.shape
    
    if n is None:
        # display all the lines, that survive the thresholding
        # of the cells of the accumulator array
        acc_maxima = normal_thresholding(acc_maxima, tsh_final)
        indices_separate = np.where(acc_maxima > 0)
        indices = list(zip(indices_separate[0], indices_separate[1]))

        for rho_on_graph, theta_i in indices:
            rho = get_rho_back(rho_on_graph, acc_r, r, c)
            theta = get_theta_back(theta_i, acc_c)
            draw_line(rho, theta, r, c, axis)
            
        image_mask = normal_thresholding2(image_gray, tsh_find_edges)
        axis.imshow(image_mask, cmap="gray")
    else:
        # display n lines that have the highest value
        # in the accumulator array (are the most sure)
        acc_maxima2 = acc_maxima.copy()
        for _ in range(n):
            max_index = np.argmax(acc_maxima2)
            max_index_2d = np.unravel_index(max_index, acc_maxima2.shape)
            rho = get_rho_back(max_index_2d[0], acc_r, r, c)
            theta = get_theta_back(max_index_2d[1], acc_c)
            acc_maxima2[max_index_2d] = 0
            #print(f"rho={rho}, theta={theta}")
            draw_line(rho, theta, r, c, axis, linewidth=1)

    if image_name is not None:
        axis.set_title(image_name)
    
    return acc

def case3d():
    _, axes = plt.subplots(1, 3)
    
    synthetic = np.zeros((100, 100))
    synthetic[10, 10] = 1
    synthetic[10, 20] = 1
    draw_lines_for_image(synthetic, axes[0], "Synthetic", 
                         1, tsh_final=0.80)
    
    oneline = to_gray(imread("images/oneline.png"))
    draw_lines_for_image(oneline, axes[1], "Oneline", 
                         1, tsh_final=1.00)

    rectangle = to_gray(imread("images/rectangle.png"))
    draw_lines_for_image(rectangle, axes[2], "Rectangle", 
                         1, tsh_final=0.35)

    plt.show()

 
#3e
def case3e():
    num_bins_theta = 300
    num_bins_rho = 300
    bricks = imread("images/bricks.jpg")
    pier = imread("images/pier.jpg")
    
    _, axes = plt.subplots(2, 2)
    axes[1, 0].imshow(bricks)
    axes[1, 1].imshow(pier)
    bricks_acc = draw_lines_for_image(to_gray(bricks), 
                                      axes[1, 0], 
                                      sigma=3, 
                                      num_bins_theta=num_bins_theta,
                                      num_bins_rho=num_bins_rho,
                                      tsh_find_edges=0.6,
                                      n=10)
    pier_acc = draw_lines_for_image(to_gray(pier), 
                                      axes[1, 1], 
                                      sigma=3, 
                                      num_bins_theta=num_bins_theta,
                                      num_bins_rho=num_bins_rho,
                                      tsh_find_edges=0.3,
                                      n=10)
    axes[0, 0].imshow(bricks_acc)
    axes[0, 0].set_title('bricks.jpg')
    axes[0, 1].imshow(pier_acc)
    axes[0, 1].set_title('pier.jpg')
    plt.show()


#3f
def lines_through_point_fixed_theta(x: int, y: int, acc, D, angle):
    # map theta = -pi/2 to -num_bins/2 and pi/2 to num_bins_2 linearly
    r, c, *_ = acc.shape
    
    # changing the value of angle on the interval [-pi, pi] to
    # a value of theta on the interval [-pi/2, pi/2]
    border = np.pi / 2
    if angle < -border:
        theta = border - (-border - angle) # wrap (underflow)
    elif angle > border:
        theta = -border + (angle - border) # wrap (overflow)
    else:
        theta = angle # no wrap (already ok value)
    
    # normalizing theta 
    theta_min = -border
    theta_max = border
    #theta_on_graph_range = c
    theta_on_graph = int((theta - theta_min) / (theta_max - theta_min) * (c - 1))
    
    # represents theta coordinate (x coordinate), 
    # which is a whole number
    rho = x * np.cos(theta) + y * np.sin(theta)

    # normalizing rho to interval [0, r], so that the 
    # lines fall inside of the graph
    rho_min = -D
    rho_max = D
    #rho_on_graph_range = r
    rho_on_graph = int((rho - rho_min) / (rho_max - rho_min) * (r - 1))
                
    acc[rho_on_graph, theta_on_graph] += 1

def case3f():
    num_bins_theta = 400
    num_bins_rho = 400
    rectangle = imread("images/rectangle.png")
    rectangle_gray = to_gray(rectangle)
    rectangle_mask = normal_thresholding2(rectangle_gray, 0.7)
    
    _, axes = plt.subplots(2, 2)
    axes[1, 0].imshow(rectangle_mask, cmap="gray")
    axes[1, 1].imshow(rectangle_mask, cmap="gray")
    rec_acc_normal = draw_lines_for_image(rectangle_gray,
                                      axes[1, 0], 
                                      sigma=3, 
                                      num_bins_theta=num_bins_theta,
                                      num_bins_rho=num_bins_rho,
                                      tsh_find_edges=0.5,
                                      tsh_final=0.4)
    rec_acc_oriented = draw_lines_for_image(rectangle_gray, 
                                      axes[1, 1], 
                                      sigma=2, 
                                      num_bins_theta=num_bins_theta,
                                      num_bins_rho=num_bins_rho,
                                      tsh_find_edges=0.7,
                                      tsh_final=0.2, 
                                      use_angles=True)
    axes[0, 0].imshow(rec_acc_normal)
    axes[0, 0].set_title('normal')
    axes[0, 1].imshow(rec_acc_oriented)
    axes[0, 1].set_title('orientation')
    plt.show()


#3g
def circles_through_point(x, y, acc, D, radius):
    # map theta = -pi/2 to -num_bins/2 and pi/2 to num_bins_2 linearly
    r, c, *_ = acc.shape
    '''this will be an array of zeroes and ones and this array will then
    be added to the input array'''
    local_array = np.zeros((r, c), dtype=np.int8)
    theta_increment = 2 * np.pi / 360 # it tracks each degree
    for theta in np.arange(-np.pi, np.pi, theta_increment):
        a = int(x - radius * np.cos(theta))
        b = int(y + radius * np.sin(theta))
        
        # normalizing rho to interval [0, r], so that the lines fall inside
        # of the graph
        #rho_on_graph = int((rho - (-D)) / (D - (-D)) * r)
        if a >= 0 and a < r and b >= 0 and b < c:
            local_array[a, b] = 1            
    acc += local_array

def draw_circle(axis, x, y, radius):
    x_coord = []
    y_coord = []
    change_scales = 2 * np.pi / 180
    for angle in range(180):
        theta = angle * change_scales
        x_coord.append(x + np.cos(theta) * radius)
        y_coord.append(y + np.sin(theta) * radius)
    axis.plot(x_coord, y_coord, color="green")

def case3g():
    radius = 47
    eclipse = imread("images/eclipse.jpg")
    eclipse_gray = to_gray(eclipse)
    #this is the first threshold
    first_threshold = 0.08
    eclipse_acc = hough_find_lines(500, 700, eclipse_gray, 1,
                                   first_threshold, radius = radius)
    eclipse_acc_maxima = nonmaxima_suppression_box(eclipse_acc)
    # this is the second threshold
    second_threshold = 0.287
    eclipse_acc_final = normal_thresholding(eclipse_acc_maxima,
                                            second_threshold)
    
    indices_separate = np.where(eclipse_acc_final > 0)
    indices = list(zip(indices_separate[0], indices_separate[1]))
    
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(eclipse_acc)
    axes[1].imshow(eclipse)
    for i in range(len(indices)):
        draw_circle(axes[1], indices[i][0], indices[i][1], radius) 
    plt.show()


#3h
def dim_norm_acc_simple(acc, image_r, image_c):
    if image_r == image_c: 
        return acc
    
    acc_fact = np.zeros(acc.shape)
    theta_dim_size = acc.shape[0]
    half_theta_dim_size = int(theta_dim_size / 2)
    even_theta_dim_size = theta_dim_size % 2 == 0
    if image_r < image_c:
        # prioritize vertical lines (angles closer to 
        # abs(theta) = 0)
        inc = (image_c / image_r - 1) / half_theta_dim_size
        value = 1
        for j in range(theta_dim_size):
            acc_fact[:, j] = value
            if even_theta_dim_size and j == half_theta_dim_size - 1:
                value = value
            else:
                if j < half_theta_dim_size:
                    value += inc
                else:
                    value -= inc
    else:
        # prioritize horizontal lines (angles closer to 
        # abs(theta) = pi/2)
        inc = (image_r / image_c - 1) / half_theta_dim_size
        value = image_r / image_c
        for j in range(theta_dim_size):
            acc_fact[:, j] = value
            if even_theta_dim_size and j == half_theta_dim_size - 1:
                value = value
            else:
                if j < half_theta_dim_size:
                    value -= inc
                else:
                    value += inc

    print(acc_fact)
    return acc * acc_fact

def sorting_points(e):
    return e[0] # return the x coordinate of the point

def distance_between_middle_two_points(point1, point2, 
                                       point3, point4):
    arr = [point1, point2, point3, point4]
    arr.sort(key=sorting_points)
    p1 = arr[1]
    p2 = arr[2]
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        
# calculates the k and n for the linear function that represents
# the line at some tho and theta, then calculates all 4 intersections
# of the enlengthened image edges with the current line and 
# calculates the Euclidean distance between the middle two points
# of the 4 intersections and divides the cel of the accumulator
# array with this calculated distance if the value in the 
# accumulator array is larger than 0, because 0 / c = 0
def dim_norm_acc(acc, image_r, image_c):
    acc_r, acc_c, *_ = acc.shape
    r = image_r
    c = image_c
    
    for j in range(acc_c):
        theta = get_theta_back(j, acc_c)
        if abs(theta) == np.pi / 2:
            acc[:, j] /= image_c
            continue
        if theta == 0:
            acc[:, j] /= image_r
            continue
            
        k = np.tan(np.pi / 2 - theta)
        for i in range(acc_r):
            if acc[i, j] > 0:
                rho = get_rho_back(i, acc_r, r, c)
                if rho == 0:
                    acc[i, j] = 0
                    continue
                
                n = - rho / np.sin(theta)
                # intersection with x = 0
                intersection1 = (0, n)
                # intersection with x = c
                intersection2 = (c, k * c + n)
                # intersection with y = 0
                intersection3 = (-n / k, 0)
                # intersection with y = -r
                intersection4 = ((-r - n) / k, -r)

                # find out, which are the middle two points
                # and calculate a distance between them
                dist = distance_between_middle_two_points(
                            intersection1,
                            intersection2,
                            intersection3,
                            intersection4)
                acc[i, j] /= dist
    
    return normalizeValues(acc)

    
def image_to_lines(image, bin_size_rows, bin_size_columns,
                       sigma=1, tsh_before=0.3, use_angles=False,
                       tsh_after=0.7, axis=plt):
    # you must change the accumulator array returned by function
    # hough_find_lines so that it favoures lines that are parralel
    # to the shorter dimension of the image and then you put that
    # transformed accumulator array in nonmaxima_suppression_box
    # function, which calculates local maxima
    r, c, *_ = image.shape
    image_gray = to_gray(image)
    image_acc = hough_find_lines(bin_size_columns, bin_size_rows,
                                 image_gray, 
                                 sigma=sigma,
                                 tsh=tsh_before,
                                 use_angles=use_angles)
    image_acc_dim_normed = dim_norm_acc(image_acc, r, c)
    image_acc_maxima = nonmaxima_suppression_box(image_acc_dim_normed)
    acc_final =  normal_thresholding(image_acc_maxima, 
                                          threshold=tsh_after)

    indices_separate = np.where(acc_final > 0)
    indices = list(zip(indices_separate[0], indices_separate[1]))

    axis.imshow(image)
    acc_r, acc_c, *_ = acc_final.shape
    for rho_on_graph, theta_i in indices:
        rho = get_rho_back(rho_on_graph, acc_r, r, c)
        theta = get_theta_back(theta_i, acc_c)
        draw_line(rho, theta, r, c, axis)
    plt.show()
    

def case3h():
    rectangle = imread('images/rectangle.png')
    image_to_lines(rectangle, 200, 200, tsh_after=0.4)
    
    pier = imread('images/pier.jpg')
    image_to_lines(pier, 400, 400)
    

# this is where you run cases:

#case1b()
#case1c()
#case1d()
#case1e()

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
case3h()