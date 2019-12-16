import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math

# img = cv.imread('./input_1.png')
# img_gs = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def cal_lbp(img_gs, radius, neighbours):
    dst = np.zeros([img_gs.shape[0] - 2*radius, img_gs.shape[1] - 2*radius], np.int)
    for k in range(neighbours):
        # neighbours corresponds to the number of nighbours we choose to calaculte the circle LBP
        # first we calculte the offset of the current pixel to the center pixel
        rx = np.float(radius * math.cos(2.0 * np.pi * k / neighbours))
        ry = np.float(radius * math.sin(2.0 * np.pi * k / neighbours))

        # double linear interpolation to calculate the gray-scale of the current "pixel"
        x1 = np.int(math.floor(rx))
        x2 = np.int(math.ceil(rx))
        y1 = np.int(math.floor(ry))
        y2 = np.int(math.ceil(ry))

        # project the offset into the interval [0, 1]
        tx = rx - x1
        ty = ry - y1

        # calculate weight for a internal "pixel"
        w1 = (1-tx) * (1-ty)
        w2 = tx * (1-ty)
        w3 = (1-tx) * ty
        w4 = tx * ty

        # we calculate the LBP value of each pixel
        for i in range(radius, img_gs.shape[0]-radius):
            for j in range(radius, img_gs.shape[1]-radius):
                center_pix = img_gs[i, j]
                current_gs = img_gs[i+x1, j+y1]*w1 + img_gs[i+x1, j+y2]*w2 + img_gs[i+x2, j+y1]*w3 + img_gs[i+x2, j+y2]*w4
                dst[i-radius, j-radius] |= (current_gs > center_pix) << (neighbours-k-1)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            current_gs = dst[i, j]
            min_gs = current_gs
            for k in range(neighbours):
                temp_gs = (current_gs >> (neighbours-k)) | (current_gs << k)
                if temp_gs < min_gs:
                    min_gs = temp_gs
            dst[i, j] = min_gs
    return dst

def proc_main(O_IN):
    s_img_url_a = O_IN["s_img_url_a"]
    s_img_url_b = O_IN["s_img_url_b"]

    img_1 = cv.imread(s_img_url_a)
    img_gs1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)

    img_2 = cv.imread(s_img_url_b)
    img_gs2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    dst1 = cal_lbp(img_gs1, 1, 8)
    hist1 = np.zeros([256])
    for i in range(dst1.shape[0]):
        for j in range(dst1.shape[1]):
            hist1[dst1[i, j]] += 1
    hist1 = hist1 / (dst1.shape[0]*dst1.shape[1])
    print("histogram_LBP input image:", hist1)
    print(sum(hist1))

    dst2 = cal_lbp(img_gs2, 1, 8)
    hist2 = np.zeros([256])
    for i in range(dst2.shape[0]):
        for j in range(dst2.shape[1]):
            hist2[dst2[i, j]] += 1
    hist2 = hist2 / (dst2.shape[0] * dst2.shape[1])
    print("histogram_LBP generated image:", hist2)

    f_img_sim = np.inner(hist1, hist2) / 0.06045487200160946233973
    #/ 0.06045487200160946233973 radius = 1
    #/ 0.03185515380920886374660 radius = 3
    print("%.23f" % f_img_sim)

    return f_img_sim

# plt.figure()
# plt.subplot(121)
# plt.imshow(img_gs, cmap='gray')
# plt.subplot(122)
# plt.imshow(dst, cmap='gray')
#
# plt.show()
