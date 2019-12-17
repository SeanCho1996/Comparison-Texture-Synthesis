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
                current_gs = img_gs[i+x1, j+y1]*w1 + img_gs[i+x1, j+y2]*w2 + img_gs[i+x2, j+y1]*w3 + img_gs[i+x2, j+y2]*w4 # interpolation for the pixels on the circle
                dst[i-radius, j-radius] |= (current_gs > center_pix) << (neighbours-k-1) # very tricky!! using bitwise operation to calculate the sum of all neighbours

    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            current_gs = dst[i, j]
            min_gs = current_gs
            for k in range(neighbours):
                temp_gs = (current_gs >> (neighbours-k)) | (current_gs << k)  # again using bitwise operation to calculate the rotation-invariant LBP
                if temp_gs < min_gs:
                    min_gs = temp_gs
            dst[i, j] = min_gs
    return dst

def lbp_hist(img, piece_number):
    stride1 = img.shape[0] // piece_number  # we cut the image into piece_number*piece_number pieces

    # initialize the final LBP_histogram vector
    final_lbp = np.array([])

    # calculate the histogram LBP in every small piece
    for i in range(piece_number):
        for j in range(piece_number):
            img_cut = img[i * stride1:(i + 1) * stride1, j * stride1:(j + 1) * stride1]
            lbp_cut = cal_lbp(img_cut, 1, 8)
            hist = np.zeros([256])
            for m in range(lbp_cut.shape[0]):
                for n in range(lbp_cut.shape[1]):
                    hist[lbp_cut[m, n]] += 1
            hist = hist / (lbp_cut.shape[0] * lbp_cut.shape[1])
            final_lbp = np.append(final_lbp, hist)

    return final_lbp

def proc_main(O_IN):
    s_img_url_a = O_IN["s_img_url_a"]
    s_img_url_b = O_IN["s_img_url_b"]

    img_1 = cv.imread(s_img_url_a)
    img_gs1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)

    img_2 = cv.imread(s_img_url_b)
    img_gs2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    hist1 = lbp_hist(img_gs1, 4)  # here we divide the image into 4*4 = 16 pieces
    hist1 = hist1 / 16  # normalization, in every small pices the sum = 1, so in total 16*1 = 16
    print("histogram_LBP input image:", hist1)
    print(np.array(hist1).shape)
    print(sum(hist1))

    hist2 = lbp_hist(img_gs2, 4)
    hist2 = hist2 / 16
    print("histogram_LBP generated image:", hist2)
    print(np.array(hist2).shape)
    # print(sum(hist2))

    f_img_sim = 0
    for i in range(hist1.shape[0]):
        dist = np.sqrt(hist1[i] * hist2[i])
        f_img_sim += dist

    print("%.23f" % f_img_sim)

    return f_img_sim

# plt.figure()
# plt.subplot(121)
# plt.imshow(img_gs, cmap='gray')
# plt.subplot(122)
# plt.imshow(dst, cmap='gray')
#
# plt.show()
