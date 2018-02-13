import imageio
from scipy.misc import imsave, imread
import cv2
import numpy as np


def resize(img, max_dim=720):
    if img.shape[0] < img.shape[1]:
        new_size = (max_dim, int(img.shape[0] / img.shape[1] * max_dim))
    else:
        new_size = (int(img.shape[1] / img.shape[0] * max_dim), max_dim)
    return cv2.resize(img, new_size)


def get_descriptors(img_path, return_img_dims=False):
    img = imread(img_path)
    img = resize(img)
    # detector = cv2.xfeatures2d.SIFT_create()
    detector = cv2.xfeatures2d.SURF_create(upright=True)
    kp, des = detector.detectAndCompute(img, None)
    if return_img_dims:
        return kp, des, img.shape[:2]
    else:
        return kp, des


def match_descriptors(kp1, des1, kp2, des2, ransac_thresholds=2.0):
    if True:
        matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        good = matcher.match(des1, des2)
        good = [(m.queryIdx, m.trainIdx) for m in good]
    elif False:
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        good = [(m.queryIdx, m.trainIdx) for m in good]
    else:
        good = lowe_ratio(des1, des2)

    def _fn(ransac_threshold):
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m].pt for m, n in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[n].pt for m, n in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            mask = mask.astype(np.bool)
        else:
            # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            mask = np.array([False] * len(good))
            M = None
        return mask, M

    if isinstance(ransac_thresholds, list):
        masks = []
        Ms = []
        for t in ransac_thresholds:
            mask, M = _fn(t)
            masks.append(mask)
            Ms.append(M)
        return good, masks, Ms
    else:
        mask, M = _fn(ransac_thresholds)
        return good, mask, M


def make_transition_gif(img_path1, img_path2, output_file):
    kp1, des1, (h1, w1) = get_descriptors(img_path1, return_img_dims=True)
    kp2, des2, (h2, w2) = get_descriptors(img_path2, return_img_dims=True)
    img1 = imread(img_path1, mode='RGB')
    img1 = resize(img1)
    img2 = imread(img_path2, mode='RGB')
    img2 = resize(img2)
    kp1, des1, (h1, w1) = get_descriptors(img_path1, return_img_dims=True)
    kp2, des2, (h2, w2) = get_descriptors(img_path2, return_img_dims=True)
    good, mask, M = match_descriptors(kp1, des1, kp2, des2, 2.0)

    img3 = cv2.warpPerspective(img1, M, img2.shape[:2][::-1])
    images = []
    for lamb in np.concatenate([np.arange(0, 1, 0.1), np.arange(1, 0, -0.1)]):
        images.append((img2 * lamb + img3 * (1 - lamb)).astype(np.uint8))
    imageio.mimsave(output_file, images, format='GIF')
