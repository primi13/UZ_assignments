from UZ_utils import *
import math


'''because np.floor(0.03 / 0.05) = 5, which is wrong, because 0.03 / 0.05
=5.999999999999999, because of awkward bit representation of numbers'''
def round_down(num):
    if abs(round(num) - num) < 0.000001:
        whole = round(num)
    else:
        whole = math.floor(num)
    return np.int32(whole)

#1
U_name = 'images/umbrellas.jpg'

I = np.array([])
I_gray = np.array([])


#1a
def case1a():
    global I
    if I.size == 0:
        I = imread(U_name)
    imshow(I)



#1b
def to_gray(arr):
    arr_new = np.copy(arr)
    for y in range(len(arr_new)):
        for x in range(len(arr_new[y])):
            arr_new[y, x] = np.mean(arr_new[y, x])
    return arr_new[:,:,0]

def case1b():
    global I, I_gray
    if I.size == 0:
        I = imread(U_name)
    if I_gray.size == 0:
        I_gray = to_gray(I)
    imshow(I_gray)

#1c
''' Answer: To correctly map scalar data in an image array to colors.
'''
def show_two(image1, image2, cmap1 = None, cmap2 = None,
             vmin1 = None, vmin2 = None, vmax1 = None, 
             vmax2 = None, title1 = False, title2 = False):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image1, cmap=cmap1, vmin = vmin1, vmax = vmax1)
    axes[1].imshow(image2, cmap=cmap2, vmin = vmin2, vmax = vmax2)
    if title1:
        axes[0].set_title(title1)
    if title2:
        axes[1].set_title(title2)
    plt.tight_layout()
    plt.show()
    return fig, axes

def case1c():
    global I, I_gray
    if I.size == 0:
        I = imread(U_name)
    if I_gray.size == 0:
        I_gray = to_gray(I)
    I_cut = I[100:200, 200:400, 0]
    show_two(I_gray, I_cut, "gray", "gray")

#1d
''' Answer: Inverting a grayscale is defined as changing
    the brightness so that the brightest parts become
    the darkest and vice versa, this is done by
    1 - I, which substracts every value in I from
    1. This works, because it is a numpy array.
'''
def case1d():
    global I
    if I.size == 0:
        I = imread(U_name)
    I_inverted = np.copy(I)
    min_row = 130
    max_row = 230
    min_col = 230
    max_col = 410
    for y in range(len(I)):
        for x in range(len(I[y])):
            if y >= min_row and y <= max_row \
            and x >= min_col and x <= max_col:
                I_inverted[y, x] = 1 - I_inverted[y, x]
    imshow(I_inverted)

#1e
def case1e():
    global I, I_gray
    if I.size == 0:
        I = imread(U_name)
    if I_gray.size == 0:
        I_gray = to_gray(I)
    I_reducted = I_gray * 0.3
    show_two(I_gray, I_reducted, "gray", "gray", 0, 0, 1, 1)


# ----------------------------------------------------------------------------


#2
B_name = 'images/bird.jpg'
B = np.array([])
B_gray = np.array([])
tsh_hand = 0.19608
B_mask2 = np.array([])
num_bins_2_1 = 30
num_bins_2_2 = 100
num_bins_2_3 = 256
B_hist = np.array([])
tsh_otsu = -1
B_otsu = np.array([])

#2a
def to_mask(arr_grey, tsh):
    arr = np.copy(arr_grey)
    arr[arr < tsh] = 0
    arr[arr >= tsh] = 1
    return arr.astype(np.uint8)

def case2a():
    global B, B_gray, B_mask2
    if B.size == 0:
        B = imread(B_name)
    if B_gray.size == 0:
        B_gray = to_gray(B)

    B_mask1 = np.where(B_gray < tsh_hand, 0, 1).astype(np.uint8)
    if B_mask2.size == 0:
        B_mask2 = to_mask(B_gray, tsh_hand)
    show_two(B_mask1, B_mask2, "gray", "gray",
            title1="np.where(...)", 
            title2="A[A < threshold]...")

#2b
'''The values or the areas of bins of the histogram are
normalized by the sum of the values so that each
value/bin represents the relative frequency or probability.
'''
def myhist(gray_image, num_bins, mymin = 0, mymax = 1):
    myrange = mymax - mymin
    bin_size = myrange / num_bins
    H = np.zeros(num_bins)
    count = 0
    for y in range(len(gray_image)):
        for x in range(len(gray_image[y])):
            index = round_down((gray_image[y, x] - mymin) / bin_size)
            if index == num_bins:
                index -= 1            
            H[index] += 1
            count += 1
    return H / count

def case2b():
    global B, B_gray
    if B.size == 0:
        B = imread(B_name)
    if B_gray.size == 0:
        B_gray = to_gray(B)
    hist1 = myhist(B_gray, num_bins_2_1)
    hist2 = myhist(B_gray, num_bins_2_2)
    hist3 = myhist(B_gray, num_bins_2_3)
    *_, axes = plt.subplots(1, 3)

    axes[0].bar(range(num_bins_2_1), hist1)
    axes[0].set_title('Number of bins: ' + str(num_bins_2_1))
    axes[1].bar(range(num_bins_2_2), hist2)
    axes[1].set_title('Number of bins: ' + str(num_bins_2_2))
    axes[2].bar(range(num_bins_2_3), hist3)
    axes[2].set_title('Number of bins: ' + str(num_bins_2_3))
    plt.show()

#2c
def myhist_mod(gray_image, num_bins):
    return myhist(gray_image, num_bins, np.min(gray_image), np.max(gray_image))

def case2c():
    global B, B_gray, B_hist
    if B.size == 0:
        B = imread(B_name)
    if B_gray.size == 0:
        B_gray = to_gray(B)
    if B_hist.size == 0:
        B_hist = myhist(B_gray, num_bins_2_3)
    B_hist_mod = myhist_mod(B_gray, num_bins_2_3)
    fig, axes = plt.subplots(1, 2)
    axes[0].bar(range(num_bins_2_3), B_hist)
    axes[0].set_title('Histogram using normal function \'myhist\'.')
    axes[1].bar(range(num_bins_2_3), B_hist_mod)
    axes[1].set_title('Histogram using modded function \'myhist_mod\'.')
    plt.tight_layout()
    plt.show()
    #this is a histogram that shows the difference of value in each bin:
    hist_diff = B_hist - B_hist_mod
    plt.bar(range(num_bins_2_3), hist_diff)
    plt.title('Histogram using \'myhist\' - \'myhist_mod\'.')
    plt.show()

#2d
def case2d():
    fig, axes = plt.subplots(2, 3)
    my_num_bins = [30, 150]
    my_names = ["moja1", "moja2", "moja3"]
    my_images = []
    for i in range(3):
        my_images.append(imread('images/' + my_names[i] + ".png"))

    for i in range(2):
        for j in range(3):
            my_image_hist = myhist(to_gray(my_images[j]), my_num_bins[i])
            axes[i, j].bar(range(my_num_bins[i]), my_image_hist)
    axes[0, 0].set_title('Histogram of the darkest image (lights off).')
    axes[0, 1].set_title('Histogram of a medium bright image (lights on barely).')
    axes[0, 2].set_title('Histogram of the brighest image (lights on fully).')
    plt.tight_layout()
    plt.show()

#2e
def init_sum_bg(hist):
    sum_bg = 0
    for i in range(len(hist)):
        sum_bg += hist[i] * i
    return sum_bg

def otsu(hist):
    n_fg = 0 # fg = foreground
    n_bg = 1 # bg = background
    sum_fg = 0
    sum_bg = init_sum_bg(hist)
    max_variance = 0
    for tsh in range(1, 257):
        curr_prob = hist[tsh - 1]
        n_fg += curr_prob
        n_bg -= curr_prob
        curr_val = tsh - 1 
        sum_fg += curr_prob * curr_val
        sum_bg -= curr_prob * curr_val
        if sum_fg == 0:
            mean_fg = 0
        else:
            mean_fg = sum_fg / n_fg
        if sum_bg == 0:
            mean_bg = 0
        else:
            mean_bg = sum_bg / n_bg
        variance = n_fg * n_bg * ((mean_fg - mean_bg) ** 2)
        if variance > max_variance:
            max_variance = variance
            max_tsh = tsh
    return max_tsh / 256

def case2e():
    global B, B_gray, B_mask2, tsh_otsu, B_hist, B_otsu
    if B.size == 0:
        B = imread(B_name)
    if B_gray.size == 0:
        B_gray = to_gray(B)
    if B_mask2.size == 0:
        B_mask2 = to_mask(B_gray, tsh_hand)
    if B_hist.size == 0:
        B_hist = myhist(B_gray, num_bins_2_3)
    if tsh_otsu == -1:
        tsh_otsu = otsu(B_hist)
    if B_otsu.size == 0:
        B_otsu = to_mask(B_gray, tsh_otsu)

    # shows the difference between otsu's and manually set threshold
    show_two(B_otsu, B_mask2, "gray", "gray", 
            title1="With Otsu:", title2="With manual thresholding:")
    *_, axes = plt.subplots(2, 3)

    # shows the otsu result for all of the given images (6)
    image_names = ['bird.jpg', 'candy.jpg', 'coins.jpg', 
                   'eagle.jpg', 'mask.png', 'umbrellas.jpg']
    for i in range(2):
        for j in range(3):
            index = 3 * i + j
            image = imread('images/' + image_names[index])
            image_gray = to_gray(image)
            image_hist = myhist(image_gray, num_bins_2_3)
            image_tsh = otsu(image_hist)
            image_mask = to_mask(image_gray, image_tsh)
            axes[i, j].imshow(image_mask, cmap='gray')
            axes[i, j].set_title(image_names[index])
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------


#3

#3a
''' 
    Answer: Based on the results, opening is erosion and then dilation, because it can
    be seen that it erases smaller dots and thin lines.
    Closing is dilation and then erosion, because it can be seen that it closes
    holes in objects.
'''
def show3a(image1, image2, image3, image4, title1, title2, title3, title4):
    *_, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(image1, cmap="gray")
    axes[0, 0].set_title(title1)
    axes[0, 1].imshow(image2, cmap="gray")
    axes[0, 1].set_title(title2)
    axes[1, 0].imshow(image3, cmap="gray")
    axes[1, 0].set_title(title3)
    axes[1, 1].imshow(image4, cmap="gray")
    axes[1, 1].set_title(title4)
    plt.tight_layout()
    plt.show()

def case3a():
    M = imread('images/mask.png')
    n1 = 5
    SE = np.ones((n1, n1))
    M_eroded1 = cv2.erode(M, SE)
    M_opened1 = cv2.dilate(M_eroded1, SE)
    M_dilated1 = cv2.dilate(M, SE)
    M_closed1 = cv2.erode(M_dilated1, SE)
    
    n2 = 10
    SE = np.ones((n2, n2))
    M_eroded2 = cv2.erode(M, SE)
    M_opened2 = cv2.dilate(M_eroded2, SE)
    M_dilated2 = cv2.dilate(M, SE)
    M_closed2 = cv2.erode(M_dilated2, SE)

    show3a(M_eroded1, M_eroded2, M_dilated1, M_dilated2,
           "eroded with square length 5", "eroded with square length 10",
           "dilated with square length 5", "dilated with square length 10")
    show3a(M_opened1, M_opened2, M_closed1, M_closed2,
           "opened with square length 5", "opened with square length 10",
           "closed with square length 5", "closed with square length 10")

#3b
def case3b():
    global B, B_gray, B_hist, tsh_otsu, B_otsu
    n = 24
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n)) 
    if B.size == 0:
        B = imread(B_name)
    if B_gray.size == 0:
        B_gray = to_gray(B)
    if B_hist.size == 0:
        B_hist = myhist(B_gray, num_bins_2_3)
    if tsh_otsu == -1:
        tsh_otsu = otsu(B_hist)
    if B_otsu.size == 0:
        B_otsu = to_mask(B_gray, tsh_otsu)
    B_dilated = cv2.dilate(B_otsu, SE)
    B_closed= cv2.erode(B_dilated, SE)
    show_two(B_otsu, B_closed, "gray", "gray", 
            title1="otsu", title2="otsu + closed")

#3c
def immask(image, mask):
    expanded_mask = np.expand_dims(mask, axis=2)
    return image * expanded_mask

# the function immask is above here, however this case is used
# to show that this function works, it is done with Otsu thresholded mask 
def case3c():
    global B, B_gray, B_hist, tsh_otsu, B_otsu
    if B.size == 0:
        B = imread(B_name)
    if B_gray.size == 0:
        B_gray = to_gray(B)
    if B_hist.size == 0:
        B_hist = myhist(B_gray, num_bins_2_3)
    if tsh_otsu == -1:
        tsh_otsu = otsu(B_hist)
    if B_otsu.size == 0:
        B_otsu = to_mask(B_gray, tsh_otsu)
    imshow(immask(B, B_otsu))

#3d
'''
The background is included in the mask and not the object, because the background is
lighter in this image eagle.jpg, so when making the image gray, the background will still
be brighter and when making the mask, the brighter parts will converge into 1 or white and
the darker parts into 0 or black, that is why the background is white and the object is black.

We could solve this with inverting the gray image, which makes the brighter parts darker and
darker parts brighter, so then the object would be white and the background black, which is
what we want. We could do this detection for needing invertion automatically with a program,
namely the program should somehow decide if the darker parts of the image are more imporant
to the viewer or if the brighter parts are and that depends on the domain. However, if we take
pictures on a bright background, we can just set all of the images to invert.

But this problem cannot be solved with a program and an image only, because the foreground can
be defined as what is important to us. So on the same image, sometimes the object is the
foreground and sometimes everything else is the foreground, because that is what interests us
at that point. For example, if we take a picture of a model with holes in it, sometimes the holes
are interesting to us and sometimes the model, which is everything except the holes.
'''
def invert_gray(gray_image):
    return 1 - gray_image

def image_to_mask(image, inverting = False):
    gray_image = to_gray(image)
    if inverting:
        gray_image = invert_gray(gray_image)
    hist = myhist(gray_image, num_bins_2_3)
    tsh = otsu(hist)
    return to_mask(gray_image, tsh)

def case3d():
    E = imread('images/eagle.jpg')

    imshow(immask(E, image_to_mask(E)))

    #fixed so that the eagle is included in the mask, not the background
    '''
    imshow(E)
    E_gray = to_gray(E)
    imshow(E_gray)
    E_inverted_gray = invert_gray(E_gray)
    imshow(E_inverted_gray)
    hist3d = myhist(E_inverted_gray, num_bins_2_3)
    tsh3d = otsu(hist3d)
    E_mask = to_mask(E_inverted_gray, tsh3d)
    imshow(E_mask)
    n = 10
    SE3d = np.ones((n,n))
    E_dilated = cv2.dilate(E_mask, SE3d)
    E_closed= cv2.erode(E_dilated, SE3d)
    imshow(E_closed)
    E_blacked = immask(E, E_closed)
    imshow(E_blacked)
    '''
#3e
def remove_label_from_mask(image, label):
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y, x] == label:
                image[y, x] = 0

def label_to_mask(image):
    image_mask = np.copy(image)
    for y in range(len(image_mask)):
        for x in range(len(image_mask[y])):
            if image_mask[y, x] > 0:
                image_mask[y, x] = 1
    return image_mask

def case3e():
    # reading the image into a numpy array
    C = imread('images/coins.jpg')

    # creating a mask from the image
    C_mask = image_to_mask(C, True)

    # cleaning the mask with morphological operations
    n = 9
    SE = np.ones((n, n))
    C_dilated = cv2.dilate(C_mask, SE)
    C_closed= cv2.erode(C_dilated, SE)

    # labeling the regions and removing those, whose area is larger than 700 pixels
    res = cv2.connectedComponentsWithStats(C_closed)
    label_image = res[1]
    label_stats = np.array(res[2])[:, 4]
    for i in range(1, len(label_stats)):
        if label_stats[i] > 700:
            remove_label_from_mask(label_image, i)

    # turning the labeled image back into a mask
    C_mask = label_to_mask(label_image)

    # with the mask change the original image
    C_mask_e = np.expand_dims(C_mask, axis=2)
    C = (C * C_mask_e) + invert_gray(C_mask_e)
    # + inverted mask is so that we change the background (which is 
    # everything but the remaining coins) from black to white

    imshow(C)



# this is where you run cases:
# (from case1a to case1e, from case2a to case2e, from case3a to case3e)

#case1a()
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

