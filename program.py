from a2_utils import *
from matplotlib import pyplot as plt
import math
import cv2
from PIL import Image
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
    horizontaly_convoluted_image = cv2.filter2D(image_gray, -1, kernel)
    return cv2.filter2D(horizontaly_convoluted_image.T, -1, kernel).T


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
    print(signal_corrupted)
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
    count = 0
    h, w, *_ = image_colored.shape
    for y in range(h):
        for x in range(w):
            indexR = round_down((image_colored[y, x, 0] - mymin) / bin_size)
            indexG = round_down((image_colored[y, x, 1] - mymin) / bin_size)
            indexB = round_down((image_colored[y, x, 2] - mymin) / bin_size)
            if indexR == num_bins:
                indexR -= 1            
            if indexG == num_bins:
                indexG -= 1            
            if indexB == num_bins:
                indexB -= 1            
            H[indexR, indexG, indexB] += 1
            count += 1
    return H / count

def case3a():
    obama = imread('images/obama.jpg')
    obama_hyst = myhist3D(obama, 16)


#3b
def euclidean(h1, h2):
    return 0

def chi_squared(h1, h2):
    return 1

def intersection(h1, h2):
    return 2

def hellinger(h1, h2):
    return 3

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
    print(compare_histograms([], [], "intersection"))

# this is where you run cases:
# (from case1b to case1e, from case2a to case2e, from case3a to case3e)

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