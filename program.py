from a2_utils import *
from matplotlib import pyplot as plt
import math
import cv2
from PIL import Image
import os
import textwrap

'''because np.floor(0.03 / 0.05) = 5, which is wrong, because 0.03 / 0.05
=5.999999999999999, because of awkward bit representation of numbers'''
def round_down(num):
    if abs(round(num) - num) < 0.000001:
        whole = round(num)
    else:
        whole = math.floor(num)
    return np.int32(whole)

# function from UZ_utils.py:
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

def norm(np_arr):
    return np_arr / np.sum(np_arr)

#1

#1b
'''
Answer:
    The shape of the kernel is that of a discrete Gaussian curve, which is
    a binomial distribution. 
    The sum of the elements in the kernel should be 1,
    because the values represent probablities. But in this case it is 0.9999999974,
    because it is a discrete sequence, not a continuous function.
    All of the numbers that the kernel overlays get sum together each with its own weight,
    but the distribution of weight among the values depends on the shape of the kernel or
    with other words the distribution of the values in the kernel.
'''
def simple_convolution(kernel_arr, signal_arr):
    # for case 1a to see if this function works:
    #kernel_arr = [0.5, 1, 0.3]
    #signal_arr = [0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0]

    N = int((len(kernel_arr) - 1)/2)
    len_I = len(signal_arr)

    signal_convoluted = np.zeros(len_I - 2 * N)

    for i in range(N, len_I - N):
        for j in range(-N, N + 1):
            signal_convoluted[i - N] += signal_arr[i - j] * kernel_arr[j + N]
    return signal_convoluted

def case1b():
    kernel_arr = read_data('kernel.txt')
    signal_arr = read_data('signal.txt')
    convoluted = simple_convolution(kernel_arr, signal_arr)
    convoluted_cv2 = cv2.filter2D(signal_arr, -1, kernel_arr)
    plt.plot(range(signal_arr.shape[0]), signal_arr, label='Original')
    plt.plot(range(kernel_arr.shape[0]), kernel_arr, label='Kernel')
    plt.plot(range(convoluted.shape[0]), convoluted, label='Result')
    plt.plot(range(convoluted_cv2.shape[0]), convoluted_cv2, label='cv2')
    plt.legend()
    plt.show()

#1c
# this function add an edge where all of the values in an added edge are the same
# and they all equal the border-most value from the original array
def add_edges(arr, padding_size):
    leftmost_value = arr[0]    
    rightmost_value = arr[len(arr) - 1]
    return np.concatenate([np.full(padding_size, leftmost_value),
                          arr,
                          np.full(padding_size, rightmost_value)])

def convolution_with_edges(kernel_arr, signal_arr):
    # for case 1a to see if this function works:
    # kernel_arr = [0.5, 1, 0.3]
    # signal_arr = [0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0]

    N = int((len(kernel_arr) - 1)/2)
    len_I = len(signal_arr)

    signal_added_edges = add_edges(signal_arr, N)
    signal_convoluted = np.zeros(len_I)
    
    # We have addes 2N values to the array, that is why now there
    #  is + N instead of - N as in the simple_convolution function.
    # The diff is len_I, so the convoluted array will be the same
    # length as the input array signal_arr.
    for i in range(N, len_I + N): 
        for j in range(-N, N + 1):
            signal_convoluted[i - N] += signal_added_edges[i - j] * kernel_arr[j + N]
    return signal_convoluted


def case1c():
    kernel_arr = read_data('kernel.txt')
    signal_arr = read_data('signal.txt')
    print(convolution_with_edges(kernel_arr, signal_arr))



#1d
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
    return whole / np.sum(whole) # normalization before return


def case1d():
    sigmas = np.array([0.5, 1, 2, 3, 4])
    w = sigmas.shape[0]

    max_modus = 0
    kernels = []
    for i in range(w):
        kernels.append(gauss(sigmas[i]))
        modus = max(kernels[i])
        if modus > max_modus:
            max_modus = modus
    max_modus *= 1.1

    # max_modus is used so that the y-axes of all of the Gaussian graphs have
    # the same limits, so that they are not changed in a way that would show
    # all Gaussian graph moduses having the same height, because they don't

    for i in range(w):
        border = math.ceil(3 * sigmas[i])
        plt.plot(range(-border, border + 1), kernels[i], label=f'sigma = {sigmas[i]}')
        plt.ylim(0, max_modus)
    plt.legend()
    plt.show()


#1e
def case1e():
    signal_arr = read_data('signal.txt')
    k1 = gauss(2)
    k2 = np.array([0.1, 0.6, 0.4])
    

    convoluted1 = convolution_with_edges(k1, signal_arr)
    convoluted11 = convolution_with_edges(k2, convoluted1)

    convoluted2 = convolution_with_edges(k2, signal_arr)
    convoluted22 = convolution_with_edges(k1, convoluted2)

    convoluted_kernel = convolution_with_edges(k2, k1)
    convoluted33 = convolution_with_edges(convoluted_kernel, signal_arr)




    *_, axes = plt.subplots(1, 4)
    xs = range(signal_arr.shape[0])
    axes[0].plot(xs, signal_arr)
    axes[0].set_title('s')
    axes[1].plot(xs, convoluted11)
    axes[1].set_title('(s * k1)  * k2')
    axes[2].plot(xs, convoluted22)
    axes[2].set_title('(s * k2)  * k1')
    axes[3].plot(xs, convoluted33)
    axes[3].set_title('s * (k1  * k2)')
    plt.show()

#2

#2a
'''
Answer:
    The Gaussian noise is better removed with the Gaussian filter.
'''
def convolute2D(image_gray, kernel):
    kernel = np.array([kernel])
    horizontaly_convoluted_image = cv2.filter2D(image_gray, -1, kernel)
    return cv2.filter2D(horizontaly_convoluted_image, -1, kernel.T)


def gaussfilter(image_gray, sigma = 1):
    kernel = gauss(sigma)
    return convolute2D(np.copy(image_gray), kernel)

def case2a():
    lena = imread('images/lena.png')
    fig, axes = plt.subplots(2, 3)

    lena_gray = to_gray(lena)
    axes[0, 0].imshow(lena_gray, cmap="gray")
    axes[0, 0].set_title('Original')

    lena_gaussian_noise = gauss_noise(lena_gray, 0.05)
    axes[0, 1].imshow(lena_gaussian_noise, cmap="gray")
    axes[0, 1].set_title('Gaussian noise')

    lena_salt_pepper = sp_noise(lena_gray, 0.05)
    axes[0, 2].imshow(lena_salt_pepper, cmap="gray")
    axes[0, 2].set_title('Salt and Pepper')

    fig.delaxes(axes[1, 0])

    lena_gaussian_noise_filtered = gaussfilter(lena_gaussian_noise)
    axes[1, 1].imshow(lena_gaussian_noise_filtered, cmap="gray")
    axes[1, 1].set_title('Filtered Gaussian noise')

    lena_salt_pepper_filtered = gaussfilter(lena_salt_pepper)
    axes[1, 2].imshow(lena_salt_pepper_filtered, cmap="gray")
    axes[1, 2].set_title('Filtered Salt and Pepper')
    plt.show()

#2b
def sharpen_image(image_gray, kernel_width):
    sharpening_kernel = - np.ones((kernel_width, kernel_width)) / (kernel_width ** 2)
    sharpening_kernel[int(kernel_width / 2), int(kernel_width / 2)] += 2                    
    museum_sharpened = cv2.filter2D(image_gray, -1, sharpening_kernel)
    return museum_sharpened


def case2b():
    kernel_width = 3
    museum = imread('images/museum.jpg')
    museum_gray = to_gray(museum)
    museum_sharpened = sharpen_image(museum_gray, kernel_width)
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(museum_gray, cmap="gray")
    axes[0].set_title('Original')
    axes[1].imshow(museum_sharpened, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title('Sharpened')
    plt.show()


#2c
'''
Answer:
    The median filter is much better at removing the salt and pepper noise 
    that the Gaussian filter. Median cannot be performed in any order and
    always get the same result. Those filters are called non-linear, because
    they are not comutative and not asociative.
'''
def simple_median(I, w):
    I_new = np.copy(I)
    len, *_ = I.shape
    arr = np.array([])
    for i in range(len):
        arr = np.append(arr, I[i])
        index_to_remove = i - w
        if index_to_remove >= 0:
            arr = np.delete(arr, 0)
        median = np.median(arr)
        I_new[i] = median
    return I_new

def case2c():
    signal = np.zeros(40)
    signal[10:20] = 1
    _, axes = plt.subplots(1, 4)
    axes[0].plot(range(40), signal)
    axes[0].set_title('Original')

    signal_corrupted = np.squeeze(sp_noise(np.expand_dims(signal, axis=1)))
    axes[1].plot(range(40), signal_corrupted)
    axes[1].set_title('Corrupted')

    signal_gauss = gaussfilter(signal_corrupted)
    axes[2].plot(range(40), signal_gauss)
    axes[2].set_title('Gauss')

    signal_gauss = simple_median(signal_corrupted, 5)
    axes[3].plot(range(40), signal)
    axes[3].set_title('Median')
    plt.show()


#2d
'''
Answer:
    Time complexity for the Gaussian filtering is O(r * c * w) and the
    time complexity for the median filtering is O(r * c * w^2 * log(w)),
    where r and c are the number of rows and columns in the image repectively
    and w is the width of the filter kernel. 
    
    However there exists an algorithm
    that in the worst case has a time complexity of O(n) for finding the k-th smallest
    number in the array of size n and the k would be round_down(len(array) / 2) to find the
    median, which we can also do in O(n), so the improved time complexity for the median
    filtering would then be O(r * c * w^2).
    
'''
def median2D(image_noise):
    image_filtered = np.copy(image_noise)
    rows, cols, *_ = image_filtered.shape
    kernel_width = 3
    kernel_half = int(kernel_width / 2)
    for i in range(rows):
        for j in range(cols):
            up = -kernel_half
            down = kernel_half
            left = -kernel_half
            right = kernel_half
            
            if i + up < 0:
                up = 0 - i
            elif i + down >= rows:
                down = rows - 1 - i
            if j + left < 0:
                left = 0 - j
            elif j + right >= cols:
                right = cols - 1 - j
            array_for_median = np.copy(
                            image_filtered[i+up : i+down+1, j+left : j+right+1])
            image_filtered[i, j] = np.median(array_for_median)
    return image_filtered

def case2d():
    lena = imread('images/lena.png')
    lena_gray = to_gray(lena)

    lena_gauss_noise = gauss_noise(lena_gray)
    lena_gauss_filtered_gauss = gaussfilter(lena_gauss_noise)
    lena_gauss_filtered_median = median2D(lena_gauss_noise)

    lena_sp_noise = sp_noise(lena_gray)
    lena_sp_filtered_gauss = gaussfilter(lena_sp_noise)
    lena_sp_filtered_median = median2D(lena_sp_noise)
    
    fig, axes = plt.subplots(2, 4)
    axes[0, 0].imshow(lena_gray, cmap="gray")
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(lena_gauss_noise, cmap="gray")
    axes[0, 1].set_title('Gaussian noise')
    axes[0, 2].imshow(lena_gauss_filtered_gauss, cmap="gray")
    axes[0, 2].set_title('Gauss filtered')
    axes[0, 3].imshow(lena_gauss_filtered_median, cmap="gray")
    axes[0, 3].set_title('Median filtered')
    fig.delaxes(axes[1, 0])
    axes[1, 1].imshow(lena_sp_noise, cmap="gray")
    axes[1, 1].set_title('Salt and pepper')
    axes[1, 2].imshow(lena_sp_filtered_gauss, cmap="gray")
    axes[1, 2].set_title('Gauss filtered')
    axes[1, 3].imshow(lena_sp_filtered_median, cmap="gray")
    axes[1, 3].set_title('Median filtered')
    plt.show()


#2e
def laplace():
    gauss_1 = gauss(50)
    n = 10
    gauss_1_2D = np.outer(gauss_1, gauss_1) * n
    h, w, *_ = gauss_1_2D.shape
    unit_impulse = np.zeros((h, w))
    unit_impulse[int(h / 2), int(w / 2)] = n - 3
    laplace = unit_impulse - gauss_1_2D
    return laplace

def laplacefilter(image_gray):
    laplace_kernel = laplace()
    return cv2.filter2D(np.copy(image_gray), -1, laplace_kernel)

    
def case2e():
    lincoln_gray = to_gray(imread('images/lincoln.jpg'))
    obama_gray = to_gray(imread('images/obama.jpg'))
    
    lincoln_gaussed = gaussfilter(lincoln_gray, 5)
    obama_laplaced = laplacefilter(obama_gray)

    coeficient_lincoln = 0.88
    coeficient_obama = 0.12
    hibrid = coeficient_lincoln * lincoln_gaussed + \
             coeficient_obama * obama_laplaced

    fig, axes = plt.subplots(2, 3)
    axes[0, 0].imshow(lincoln_gray, cmap="gray")
    axes[0, 0].set_title('Image1')
    axes[0, 1].imshow(obama_gray, cmap="gray")
    axes[0, 1].set_title('Image 2')
    axes[0, 2].imshow(hibrid, cmap="gray")
    axes[0, 2].set_title('Result')

    axes[1, 0].imshow(lincoln_gaussed, cmap="gray")
    axes[1, 0].set_title('Gauss')
    axes[1, 1].imshow(obama_laplaced, cmap="gray")
    axes[1, 1].set_title('Laplace')
    fig.delaxes(axes[1, 2])
    plt.show()



#3

#3a
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
    
    return H / np.sum(H)

def myhist1D(image_colored, num_bins):
    return myhist3D(image_colored, num_bins, 0, 1).reshape(-1)

def case3a():
    obama = imread('images/obama.jpg')
    obama_hist = myhist3D(obama, 16)


#3b
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


def case3b():
    obama = imread('images/obama.jpg')
    obama_hist = myhist3D(obama, 16)
    
    lincoln = imread('images/lincoln.jpg')
    lincoln_hist = myhist3D(lincoln, 16)
    print(compare_histograms(obama_hist, lincoln_hist, "l2"))


#3c
'''
Answer:
    Object_03_1.png (boat/toy) is closer to object_01_1.png than
    object_02_1.png (onion) according to the L2 (Euclidean) distance.
    
    According to chi-squared, intersection and Hellinger distances 
    the boat toy is also closer.
    
    The most dominant color is the background color, which is close to black.
'''
dist_measures = ["l2", "chi", "inter", "hell"]


def case3c():
    obj1 = imread('dataset/object_01_1.png')
    obj2 = imread('dataset/object_02_1.png')
    obj3 = imread('dataset/object_03_1.png')
    obj1_hist = myhist3D(obj1, 8).reshape(-1)
    obj2_hist = myhist3D(obj2, 8).reshape(-1)
    obj3_hist = myhist3D(obj3, 8).reshape(-1)

    dist_measure = dist_measures[3]
    
    dist11 = compare_histograms(obj1_hist, obj1_hist, dist_measure)
    dist12 = compare_histograms(obj1_hist, obj2_hist, dist_measure)
    dist13 = compare_histograms(obj1_hist, obj3_hist, dist_measure)

    
    *_, axes = plt.subplots(2, 3)
        
    axes[0, 0].imshow(obj1)
    axes[0, 0].set_title('Image 1')
    axes[0, 1].imshow(obj2)
    axes[0, 1].set_title('Image 2')
    axes[0, 2].imshow(obj3)
    axes[0, 2].set_title('Image 3')
    axes[1, 0].bar(range(obj1_hist.shape[0]), obj1_hist, 5)
    axes[1, 0].set_title(f'{dist_measure}(h1, h1) = {dist11:.2f}')
    axes[1, 1].bar(range(obj2_hist.shape[0]), obj2_hist, 5)
    axes[1, 1].set_title(f'{dist_measure}(h1, h2) = {dist12:.2f}')
    axes[1, 2].bar(range(obj3_hist.shape[0]), obj3_hist, 5)
    axes[1, 2].set_title(f'{dist_measure}(h1, h3) = {dist13:.2f}')
    
    plt.show()
    
#3d
'''
Answer:
    Probably the best image retrieval distance mesaure is hellinger,
    because it gave the biggest difference in distances from pictures
    that were only rotated reference images and other images.
    
    The retrieved sequence doesn't really change with the number of bins, 
    but it could if the number of bins would be really high, for example 256,
    because then the noise would play a larger role in claculating the distance
    and of course if the number of bins is too small like less then 4 or 5, then
    the algorithm is not accurate enough.
    
    The distribution of pixel values into the 3D histogram array really isn't affected
    by the number of bins, but the next operations are affecting the run time
    such as reshaping the histogram, sorting it and especially displaying it using plt.bar().
'''
def get_histograms1D(folder_path, num_bins):
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
        hist = myhist1D(image, num_bins)
        images_and_histograms.append((image_file, image, hist))
        #print(images_and_histograms)
    return images_and_histograms

def sorting_atribute(e):
    return e[3]
    
def case3d(num_bins_per_dim = 32,  weights = None):
    images_and_histograms = get_histograms1D("dataset", num_bins_per_dim)
    reference_image_and_histogram = images_and_histograms[19]
    reference_histogram = reference_image_and_histogram[2]
    if weights is not None:
        reference_histogram = weigh(reference_histogram, weights)
    
    distance_measure = dist_measures[3]

    
    
    i = 0
    for image_and_hist in images_and_histograms:
        hist = image_and_hist[2]
        if weights is not None:
            hist = weigh(hist, weights)
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
    riah = reference_image_and_histogram
    
    *_, axes = plt.subplots(2, 6)
            
    for i in range(6):
        axes[0, i].imshow(ciah[i][1])
        axes[0, i].set_title(ciah[i][0])
        axes[1, i].bar(range(ciah[i][2].shape[0]), ciah[i][2], width=10)
        axes[1, i].set_title(f"{distance_measure}={ciah[i][3]:.2f}")
    plt.show()

def distances_from_reference_array(reference_histogram, images_and_histograms, 
                                  distance_measure, weights = None):
    distances_from_reference = []
    for image_and_hist in images_and_histograms:
        hist = image_and_hist[2]
        if weights is not None:
            hist = weigh(hist, weights)
        dist = compare_histograms(reference_histogram, hist, distance_measure)
        distances_from_reference.append(dist)
    return distances_from_reference


#3e
def case3e(num_bins_per_dim = 8,  weights = None):
    images_and_histograms = get_histograms1D("dataset", num_bins_per_dim)
    reference_image_and_histogram = images_and_histograms[19]
    reference_histogram = reference_image_and_histogram[2]
    
    distances_from_reference_all = []
    distances_from_reference_sorted_all = []
    distance_mesaures = ["l2", "chi", "inter", "hell"]    
    
    # all distance measures
    for distance_mesaure in distance_mesaures:
        distances_from_reference = distances_from_reference_array(
                                    reference_histogram, 
                                    images_and_histograms, 
                                    distance_mesaure,
                                    weights)
        distances_from_reference_sorted = distances_from_reference.copy()
        distances_from_reference_sorted.sort()
        
        distances_from_reference_all.append(distances_from_reference)
        distances_from_reference_sorted_all.append(distances_from_reference_sorted)
        
    distances_from_reference_hell = distances_from_reference_all.pop()
    distances_from_reference_sorted_hell = distances_from_reference_sorted_all.pop()
    
    num_minimums = 5
    circled = distances_from_reference_sorted_hell[:num_minimums]
    circled_indexes = []
    for i, val in enumerate(circled):
        circled_indexes.append(distances_from_reference_hell.index(val))
    
    w = len(distances_from_reference)
    x = range(w)
    
    args = {'linestyle':'', 'marker': 'o', 'markerfacecolor':'none', 
            'markeredgecolor':'orange'}
    
    *_, axes = plt.subplots(1, 2)
    axes[0].plot(x, distances_from_reference_hell)
    axes[0].plot(circled_indexes, circled, **args)
    axes[0].set_title('Unsorted')
    axes[1].plot(x, distances_from_reference_sorted_hell)
    axes[1].plot(range(num_minimums), circled, **args)
    axes[1].set_title('Sorted')
    plt.show()
    

#3f
'''
    Answer:
    The prevailing color (by a lot) is a bit above (0, 0, 0), which is a
    dark grey color and that is because most images have only this color or
    something similar for the background, while other color in the foreground
    are distributed more equally and so there aren't so many outstanders.
    
    Yes, the weighing helped with retrieving the relevant result, now the
    similarity check is better.
'''
def get_only_histograms1D(folder_path, num_bins):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
        # Filter only files with image extensions
    image_files = [f for f in files if f.lower().
                   endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    histograms = []
    
    # Open and process each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        image = imread(image_path)
        hist = myhist1D(image, num_bins)
        histograms.append(hist)
    return histograms

scalling_constant = 10

# x wil be the frequency of the bin
def weigh_func(x):
    return np.exp(-scalling_constant * x) 

def weigh(np_arr, np_weights):
    return norm(np_arr * np_weights)

def get_weights(num_bins_per_dim):

    images_and_histograms = get_only_histograms1D("dataset", num_bins_per_dim)

    summed_histogram = np.zeros(num_bins_per_dim ** 3)
    for image_and_hist in images_and_histograms:
        summed_histogram += image_and_hist

    #norm = len(images_and_histograms)
    #summed_histogram /= norm
     
    #max_index = np.argmax(summed_histogram)
    #print(f"max = (x: {max_index}, y: {summed_histogram[max_index]})")
    
    w = summed_histogram.shape[0]
    plt.bar(range(w), summed_histogram, width=5)
    plt.show()
    
    weigh_func_vect = np.vectorize(weigh_func)
    weighted_histogram = weigh_func_vect(summed_histogram)
    return norm(weighted_histogram) # normalization

def case3f():
    global scalling_constant
    scalling_constant = 10
    num_bins_per_dim = 8
    weights = get_weights(num_bins_per_dim)

    case3d(num_bins_per_dim, weights)


# this is where you run cases:
# (from case1b to case1e, from case2a to case2e, from case3a to case3f)

#case1b()
#case1c()
#case1d()
#case1e()

#case2a()
#case2b()
#case2c()
#case2d()
#case2e()

#case3a()
#case3b()
#case3c()
#case3d()
#case3e()
#case3f()