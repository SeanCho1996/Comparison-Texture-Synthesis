import json
from img_gist_feature.utils_gist import *
import matplotlib.pyplot as plt
import cv2


def get_img_gist_feat(s_img_url):
    gist_helper = GistUtils()
    np_img = preproc_img(s_img_url)
    np_gist = gist_helper.get_gist_vec(np_img)
    np_gist_L2Norm = np_l2norm(np_gist).reshape(-1)
    
    print()
    print("img url: %s" % s_img_url) 
    print("gist feature:", np_gist)
    print("gist feature(L2 norm):", np_gist_L2Norm)
    print()
    #print("hfhfhskfkjs")
    #print(s_img_url.split('/'))
    return np_gist_L2Norm

def proc_main(O_IN):
    s_img_url_a = O_IN["s_img_url_a"]
    s_img_url_b = O_IN["s_img_url_b"]

    np_img_gist_a = get_img_gist_feat(s_img_url_a)
    np_img_gist_b = get_img_gist_feat(s_img_url_b)

    f_img_sim = np.inner(np_img_gist_a, np_img_gist_b)
    print("%.23f" % f_img_sim)
    # print("sum:", sum(np_img_gist_a))


    # np_img_group = cv2.imread(s_img_url_a)
    np_img_public = cv2.imread(s_img_url_b)

    return f_img_sim
    # fig = plt.figure()
    # plt.suptitle("%.7f" % f_img_sim, fontsize=20)
    #
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(np_img_group[:,:,::-1])
    # ax.set_title("input", fontsize=20)
    #
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(np_img_public[:,:,::-1])
    # ax.set_title("%s" % s_img_url_b.split('/')[3], fontsize=20)
    #
    # fig.savefig("./test/show.jpg")
    #
    # plt.show()


# if __name__ == "__main__":
#     O_IN = {}
#     O_IN['s_img_url_a'] = "./input_1.png"
#     O_IN['s_img_url_b'] = "../img_results/input_1/result_1.png"
#     proc_main(O_IN)

