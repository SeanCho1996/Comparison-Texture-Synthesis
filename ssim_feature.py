import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def conv2d(img, kernel):
    # kernel size
    k_height = kernel.shape[0]
    k_width = kernel.shape[1]

    # output img size
    res_height = img.shape[0] - k_height + 1
    res_width = img.shape[1] - k_width + 1

    result = np.zeros([res_height, res_width], dtype=np.float)

    # affectation element by element
    for i in range(res_height):
        for j in range(res_width):
            result[i][j] = (img[i:i+k_height, j:j+k_width] * kernel).sum()
    return result


def gaussian_filter(k_size=3, sigma=1.5):
    filter = np.zeros([k_size, k_size], dtype=np.float)
    center_pos = k_size//2

    # in case of sigma =0
    if sigma == 0:
        sigma = ((k_size - 1)*0.5 - 1) * 0.3 + 0.8

    # calculate each element of the filter
    s = 2*(sigma ** 2)
    for i in range(k_size):
        for j in range(k_size):
            x = i - center_pos
            y = j - center_pos
            filter[i][j] = np.exp(-(x**2 + y**2)/s)

    # normalization
    norm = sum(sum(filter))
    filter = filter/norm

    return filter


def ssim(O_IN, window_size=3, sigma=1.5, k1=0.01, k2=0.03, alpha=1, beta=1, gama=1):
    # read image
    s_img_url_a = O_IN["s_img_url_a"]
    s_img_url_b = O_IN["s_img_url_b"]

    img_1 = cv.imread(s_img_url_a)
    img_gs1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)

    img_2 = cv.imread(s_img_url_b)
    img_gs2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    # for ssim similarity, it requires the two images have the same size
    img_gs1 = np.resize(img_gs1, img_gs2.shape)

    # generate kernel
    kernel = gaussian_filter(window_size, sigma)

    # 1. lightness similarity
    ux = conv2d(img_gs1, kernel)
    uy = conv2d(img_gs2, kernel)
    uxuy = ux * uy
    ux_sqr = ux ** 2
    uy_sqr = uy ** 2

    c1 = (k1 * 255) ** 2

    L = ((2*uxuy + c1) / (ux_sqr + uy_sqr + c1)) ** alpha

    # 2. saturation similarity
    x2 = img_gs1 ** 2
    y2 = img_gs2 ** 2
    xy = img_gs1 * img_gs2

    ux2 = conv2d(x2, kernel)
    uy2 = conv2d(y2, kernel)
    uxy = conv2d(xy, kernel)

    sx_sqr = np.abs(ux2 - ux_sqr)
    sy_sqr = np.abs(uy2 - uy_sqr)
    sxy = np.abs(uxy - uxuy)

    sxsy = np.sqrt(sx_sqr)*np.sqrt(sy_sqr)

    c2 = (k2 * 255) ** 2
    C = (2*sxsy + c2) / (sx_sqr + sy_sqr + c2)
    C = C ** beta

    # 3. structure similarity
    c3 = 0.5 * c2
    S = (sxy + c3) / (sxsy + c3)
    S = S ** gama

    ssim = L * C * S

    return np.mean(ssim)


