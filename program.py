from a2_utils import *
from matplotlib import pyplot as plt
import math
import cv2
from PIL import Image


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


def gaussfilter(image_gray):
    kernel = gauss(1)
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
def case2b():
    n = 3
    sharpening_kernel = - np.ones((n, n)) / (n ** 2)
    sharpening_kernel[int(n / 2), int(n / 2)] += 2                    
    museum = imread('images/museum.jpg')
    museum_gray = to_gray(museum)
    museum_sharpened = cv2.filter2D(museum_gray, -1, sharpening_kernel)
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(museum_gray, cmap="gray")
    axes[0].set_title('Original')
    axes[1].imshow(museum_sharpened, cmap="gray")
    axes[1].set_title('Sharpened')
    plt.show()


#2c
def simple_median(I, w):
    I_new = np.copy(I)
    kernel = np.ones(w)
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
