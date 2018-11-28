import imageio
from scipy.misc import imsave, imread
import cv2
import numpy as np
from functools import lru_cache
from typing import Union, List
from collections import Iterable


def resize(img, max_dim=720):
    if img.shape[0] < img.shape[1]:
        new_size = (max_dim, int(img.shape[0] / img.shape[1] * max_dim))
    else:
        new_size = (int(img.shape[1] / img.shape[0] * max_dim), max_dim)
    return cv2.resize(img, new_size)


@lru_cache(maxsize=10000)
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


def match_descriptors(kp1, des1, kp2, des2, ransac_thresholds: Union[float, Iterable]=2.0):
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

    def _fn(ransac_threshold: float):
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m].pt for m, n in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[n].pt for m, n in good]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            mask = np.less_equal(np.linalg.norm(dst_pts[:, 0, :] - cv2.perspectiveTransform(src_pts, M)[:, 0, :], axis=1), ransac_threshold)
            #mask = mask.astype(np.bool)
        else:
            # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            mask = np.array([False] * len(good))
            M = None
        return mask, M

    if isinstance(ransac_thresholds, Iterable):
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


def get_duplicate_features(img_path1, img_path2, main_threshold=5.0, ransac_thresholds=(1.0, 2.0, 5.0, 10.0, 20.0),
                           return_plot=False, return_boxes=False, dataset=None):
    if dataset is not None:
        img_path1 = dataset.get_image_path(img_path1)
        img_path2 = dataset.get_image_path(img_path2)
    kp1, des1, (h1, w1) = get_descriptors(img_path1, return_img_dims=True)
    kp2, des2, (h2, w2) = get_descriptors(img_path2, return_img_dims=True)

    all_thresholds = [main_threshold] + (
        list(ransac_thresholds) if isinstance(ransac_thresholds, Iterable) else [ransac_thresholds])
    good, masks, Ms = match_descriptors(kp1, des1, kp2, des2, all_thresholds)
    num_matches = np.array([np.sum(mask) for mask in masks[1:]])
    matchesMask = masks[0].ravel().tolist()
    M = Ms[0]
    mask = masks[0]

    # boxes
    src_pts = np.float32([kp1[m].pt for m, n in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[n].pt for m, n in good]).reshape(-1, 1, 2)
    m1 = np.min(src_pts[mask], axis=0)
    m2 = np.max(src_pts[mask], axis=0)

    def _get_box(m1, m2, h, w):
        return {
            'x': m1[0] / w, 'y': m1[1] / h, 'w': (m2[0] - m1[0]) / w, 'h': (m2[1] - m1[1]) / h
        }

    def _get_spatial_spread(m1, m2, h, w):
        return (m2[0] - m1[0]) * (m2[1] - m1[1]) / (h * w)

    n1 = np.min(dst_pts[mask], axis=0)
    n2 = np.max(dst_pts[mask], axis=0)
    boxes = (_get_box(m1, m2, h1, w1), _get_box(n1, n2, h2, w2))
    spatial_spreads = (_get_spatial_spread(m1, m2, h1, w1), _get_spatial_spread(n1, n2, h2, w2))

    # min(#des), max(#des), #candidate_pairs, min(spatial_spread), max(spatial_spread)
    # , [proportion_good_matches for each threshold]
    features = np.concatenate([
        np.array([min(len(des1), len(des2)), max(len(des1), len(des2)), len(good),
                  min(spatial_spreads), max(spatial_spreads)]),
        num_matches / len(good)
    ])

    if return_boxes:
        return features, boxes
    if return_plot:
        if M is not None:
            m1 = np.min(src_pts[mask], axis=0)
            m2 = np.max(src_pts[mask], axis=0)
            pts = np.float32([[m1[0], m1[1]], [m1[0], m2[1] - 1], [m2[0] - 1, m2[1] - 1], [m2[0] - 1, m1[1]]]).reshape(
                -1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img1 = cv2.polylines(img1, [np.int32(pts)], True, 255, 3, cv2.LINE_AA)
            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        return features, img3
    else:
        return features


def get_aligned_images(img_path1, img_path2, return_matrix=False, threshold=15.0):
    img1 = imread(img_path1, mode='RGB')
    img1 = resize(img1)
    img2 = imread(img_path2, mode='RGB')
    img2 = resize(img2)
    kp1, des1, (h1, w1) = get_descriptors(img_path1, return_img_dims=True)
    kp2, des2, (h2, w2) = get_descriptors(img_path2, return_img_dims=True)
    good, mask, M = match_descriptors(kp1, des1, kp2, des2, threshold)

    img3 = cv2.warpPerspective(img1, M, img2.shape[:2][::-1])
    if return_matrix:
        return img3, img2, M
    else:
        return img3, img2


def is_transition_valid(img_path1, img_path2, threshold=15.0):
    img1 = imread(img_path1, mode='RGB')
    img1 = resize(img1)
    img2 = imread(img_path2, mode='RGB')
    img2 = resize(img2)
    kp1, des1, (h1, w1) = get_descriptors(img_path1, return_img_dims=True)
    kp2, des2, (h2, w2) = get_descriptors(img_path2, return_img_dims=True)
    good, mask, M = match_descriptors(kp1, des1, kp2, des2, threshold)

    return bool(np.sum(mask) > 15)


def make_transition_gif(img_path1, img_path2, output_file):
    img3, img2 = get_aligned_images(img_path1, img_path2)
    images = []
    for lamb in np.concatenate([np.arange(0, 1, 0.1), np.arange(1, 0, -0.1)]):
        images.append((img2 * lamb + img3 * (1 - lamb)).astype(np.uint8))
    imageio.mimsave(output_file, images, format='GIF')
