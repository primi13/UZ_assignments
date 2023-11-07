from a2_utils import *
from matplotlib import pyplot as plt
import math

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
    print(simple_convolution(kernel_arr, signal_arr))

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

    *_, axes = plt.subplots(1, w)
    for i in range(w):
        border = math.ceil(3 * sigmas[i])
        axes[i].bar(range(-border, border + 1), kernels[i])
        axes[i].set_ylim(0, max_modus)
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