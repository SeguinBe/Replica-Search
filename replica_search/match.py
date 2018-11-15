import numpy as np
import numba
from numba import njit
from scipy.spatial import distance
from sklearn.linear_model import RANSACRegressor
import cv2
from .utils import Timer


@njit
def normalize(f_map, norm_epsilon):
    for i in range(f_map.shape[0]):
        for j in range(f_map.shape[1]):
            f_map[i, j, :] /= np.linalg.norm(f_map[i, j]) + norm_epsilon


@njit
def nb_unravel(ind, dims):
    result = np.empty(len(dims), dtype=np.int32)
    offsets = np.empty(len(dims), dtype=np.int32)
    d = len(dims)
    o = 1
    for i in range(d):
        offsets[d-1-i] = o
        o *= dims[d-1-i]
    for i, o in enumerate(offsets):
        result[i] = ind // o
        ind = ind % o
    return result

@njit
def nb_unravel_array(inds, dims):
    result = np.empty((len(inds), len(dims)), dtype=np.int32)
    offsets = np.empty(len(dims), dtype=np.int32)
    d = len(dims)
    o = 1
    for i in range(d):
        offsets[d-1-i] = o
        o *= dims[d-1-i]
    for k in range(len(inds)):
        ind = inds[k]
        for i, o in enumerate(offsets):
            result[k, i] = ind // o
            ind = ind % o
    return result


@njit(nogil=True)
def get_candidates(f_map_1, f_map_2, norm_epsilon=0, margin=1, crosscheck_limit=2):
    normalize(f_map_1, norm_epsilon)
    normalize(f_map_2, norm_epsilon)

    h1, w1, d_size = f_map_1.shape

    h2, w2, _ = f_map_2.shape
    # Convert to descriptors, keypoint versions
    des1 = np.ascontiguousarray(f_map_1[margin:h1 - margin, margin:w1 - margin]).reshape((-1, d_size))
    des2 = np.ascontiguousarray(f_map_2[margin:h2 - margin, margin:w2 - margin]).reshape((-1, d_size))
    kp1 = (nb_unravel_array(np.arange(len(des1)), (h1 - 2 * margin, w1 - 2 * margin)) + 0.5 + margin).astype(np.float32)
    kp2 = (nb_unravel_array(np.arange(len(des2)), (h2 - 2 * margin, w2 - 2 * margin)) + 0.5 + margin).astype(np.float32)
    # Because of the margin, the arrays might be empty
    if len(kp1) == 0 or len(kp2) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32),  np.empty((0,), np.float32)
    # d = distance.cdist(des1, des2)
    d = 1 - des1 @ des2.T  # Warning, that is ~1/2 of the euclidean distance since des1.norm ~ des2.norm ~ 1
    best_1 = np.empty((len(d), crosscheck_limit), dtype=np.int32)
    for i in range(len(d)):
        best_1[i, :] = np.argsort(d[i, :])[:crosscheck_limit]
    d_T = d.T
    best_2 = np.empty((len(d_T), crosscheck_limit), dtype=np.int32)
    for i in range(len(d_T)):
        best_2[i, :] = np.argsort(d_T[i, :])[:crosscheck_limit]

    # best_1 = np.argsort(d, axis=1)[:, :crosscheck_limit]
    # best_2 = np.argsort(d.T, axis=1)[:, :crosscheck_limit]
    best_1_o = [set(best_1[i, :]) for i in range(len(best_1))]
    best_2_o = [set(best_2[i, :]) for i in range(len(best_2))]
    # d = des1 @ des2.T
    # best_1 = np.argmax(d, axis=1)
    # best_2 = np.argmax(d, axis=0)
    good = np.array([(i, j) for i, s in enumerate(best_1_o) for j in s if i in best_2_o[j]])
    src_pts = kp1[good[:, 0]]
    dst_pts = kp2[good[:, 1]]
    distances = np.array([d[good[i, 0], good[i, 1]] for i in range(len(good))])
    return src_pts, dst_pts, distances


def get_candidates_old(f_map_1, f_map_2, norm_epsilon=0, margin=1, crosscheck_limit=2):
    f_map_1 = f_map_1/(np.linalg.norm(f_map_1, axis=-1, keepdims=True) + norm_epsilon)
    h1, w1, d_size = f_map_1.shape
    f_map_2 = f_map_2/(np.linalg.norm(f_map_2, axis=-1, keepdims=True) + norm_epsilon)
    h2, w2, _ = f_map_2.shape
    # Convert to descriptors, keypoint versions
    des1 = f_map_1[margin:h1 - margin, margin:w1 - margin].reshape((-1, d_size))
    des2 = f_map_2[margin:h2 - margin, margin:w2 - margin].reshape((-1, d_size))
    kp1 = np.stack(np.unravel_index(np.arange(len(des1)), (h1 - 2 * margin, w1 - 2 * margin)), axis=1) + 0.5 + margin
    kp2 = np.stack(np.unravel_index(np.arange(len(des2)), (h2 - 2 * margin, w2 - 2 * margin)), axis=1) + 0.5 + margin

    # d = distance.cdist(des1, des2)
    d = 1 - des1 @ des2.T  # Warning, that is ~1/2 of the euclidean distance since des1.norm ~ des2.norm ~ 1
    best_1 = np.argsort(d, axis=1)[:, :crosscheck_limit]
    best_2 = np.argsort(d.T, axis=1)[:, :crosscheck_limit]
    best_1 = [set(s) for s in best_1]
    best_2 = [set(s) for s in best_2]
    # d = des1 @ des2.T
    # best_1 = np.argmax(d, axis=1)
    # best_2 = np.argmax(d, axis=0)
    good = [(i, j) for i, s in enumerate(best_1) for j in s if i in best_2[j]]
    src_pts = np.float32([kp1[m] for m, _ in good])
    dst_pts = np.float32([kp2[m] for _, m in good])
    distances = d[list(zip(*good))]
    return src_pts, dst_pts, distances


@njit(nogil=True)
def spatially_coherent_mask(src_pts, dst_pts, residual_threshold=2.0):
    min_x0_y0, increment_x0_y0 = -15, 1
    max_x0_y0 = -min_x0_y0
    possible_x0 = np.arange(min_x0_y0, max_x0_y0, increment_x0_y0, np.int32)
    possible_y0 = np.arange(min_x0_y0, max_x0_y0, increment_x0_y0, np.int32)
    possible_lambdas = np.exp(np.arange(-7, 7)*0.2)  # ln(4) ~ 1.4 so 0.25-4x zoom
    possible_params = np.zeros((len(possible_x0), len(possible_y0), len(possible_lambdas), 2))
    for i in range(len(src_pts)):
        src_y, src_x = src_pts[i]
        dst_y, dst_x = dst_pts[i]
        for i_lamb, lamb in enumerate(possible_lambdas):
            x0 = dst_x-lamb*src_x
            y0 = dst_y-lamb*src_y
            i_x0 = int(round((x0-min_x0_y0)/increment_x0_y0))
            i_y0 = int(round((y0-min_x0_y0)/increment_x0_y0))
            if 0 <= i_x0 < len(possible_x0) and 0 <= i_y0 < len(possible_y0):
                possible_params[i_x0, i_y0, i_lamb, 0] += 1

            # Flip
            x0 = dst_x+lamb*src_x
            i_x0 = int(round((x0-min_x0_y0)/increment_x0_y0))
            if 0 <= i_x0 < len(possible_x0) and 0 <= i_y0 < len(possible_y0):
                possible_params[i_x0, i_y0, i_lamb, 1] += 1

    best_inds = np.argsort(possible_params.ravel())[-5:]
    best_inliers = 0
    best_mask = np.full(len(src_pts), False, dtype=np.bool_)
    best_M = np.array([
                [1, 0],
                [0, 1],
                [0, 0]
            ], dtype=np.float32)
    mask = best_mask.copy()
    preds = np.empty((1, 2), dtype=np.float32)
    src_pts_intercept = np.concatenate((src_pts, np.ones((len(src_pts), 1), dtype=np.float32)), axis=1)
    for ind in best_inds:
        i_x0, i_y0, i_lamb, is_flipped = nb_unravel(ind, dims=possible_params.shape)
        x0 = possible_x0[i_x0]
        y0 = possible_y0[i_y0]
        lamb = possible_lambdas[i_lamb]

        M = np.array([
                [lamb, 0],
                [0, (1-2*int(is_flipped))*lamb],
                [y0, x0]
            ], dtype=np.float32)
        preds = src_pts_intercept @ M
        mask = np.sum(np.square(preds-dst_pts), axis=1) <= residual_threshold
        if np.sum(mask) > best_inliers:
            best_inliers = np.sum(mask)
            best_mask[:] = mask
            best_M = M

    assert np.sum(best_mask) == best_inliers, "weird"

    if best_inliers > 0:
        assert np.sum(best_mask) > 0, "weird2"
        # Refine matrix
        #M, _, _, _ = np.linalg.lstsq(np.concatenate([src_pts[best_mask], np.ones((np.sum(best_mask), 1))], axis=1),
        #                             dst_pts[best_mask], rcond=-1)
        return best_M, best_mask
    else:
        print("No inliers?")
        return best_M, best_mask


def match_feature_maps(f_map_1, f_map_2, norm_epsilon=0, margin=1, crosscheck_limit=3):
    with Timer("candidates", disable=True):
        src_pts, dst_pts, distances = get_candidates(f_map_1, f_map_2, norm_epsilon, margin, crosscheck_limit)

    if len(src_pts) == 0:
        print("No candidates")

    with Timer("spatially_coherent", disable=True):
        M, mask = spatially_coherent_mask(src_pts, dst_pts, residual_threshold=2.0)
    num_matches = int(np.sum(mask))

    h1, w1, _ = f_map_1.shape
    h2, w2, _ = f_map_2.shape

    if num_matches == 0:
        return num_matches, None, (src_pts, dst_pts), mask.tolist(), ((0,0,1,1), (0,0,1,1))

    m1 = np.min(src_pts[mask], axis=0)
    m2 = np.max(src_pts[mask], axis=0)
    box1 = ((m1[0]-0.5)/h1, (m1[1]-0.5)/w1, (m2[0]-m1[0]+1)/h1, (m2[1]-m1[1]+1)/w1)
    m1 = np.min(dst_pts[mask], axis=0)
    m2 = np.max(dst_pts[mask], axis=0)
    box2 = ((m1[0]-0.5)/h2, (m1[1]-0.5)/w2, (m2[0]-m1[0]+1)/h2, (m2[1]-m1[1]+1)/w2)

    return num_matches, None, (src_pts, dst_pts), mask.tolist(), (box1, box2)


def match_feature_maps_old(f_map_1, f_map_2, norm_epsilon=0, margin=1, crosscheck_limit=3, ransac_max_trials=100):
    f_map_1 /= np.linalg.norm(f_map_1, axis=-1, keepdims=True) + norm_epsilon
    h1, w1, d_size = f_map_1.shape

    f_map_2 /= np.linalg.norm(f_map_2, axis=-1, keepdims=True) + norm_epsilon
    h2, w2, _ = f_map_2.shape
    # Convert to descriptors, keypoint versions
    des1 = f_map_1[margin:h1 - margin, margin:w1 - margin].reshape((-1, d_size))
    des2 = f_map_2[margin:h2 - margin, margin:w2 - margin].reshape((-1, d_size))
    kp1 = np.stack(np.unravel_index(np.arange(len(des1)), (h1 - 2 * margin, w1 - 2 * margin)),
                   axis=1) + 0.5 + margin
    kp2 = np.stack(np.unravel_index(np.arange(len(des2)), (h2 - 2 * margin, w2 - 2 * margin)),
                   axis=1) + 0.5 + margin

    if False:
        # Lowe ratio test
        matcher = cv2.BFMatcher(crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
    else:
        d = distance.cdist(des1, des2)
        #d = 1 - des1 @ des2.T  # Warning, that is ~1/2 of the euclidean distance since des1.norm ~ des2.norm ~ 1
        best_1 = np.argsort(d, axis=1)[:, :crosscheck_limit]
        best_2 = np.argsort(d.T, axis=1)[:, :crosscheck_limit]
        best_1 = [set(s) for s in best_1]
        best_2 = [set(s) for s in best_2]
        # d = des1 @ des2.T
        # best_1 = np.argmax(d, axis=1)
        # best_2 = np.argmax(d, axis=0)
        good = [(i, j) for i, s in enumerate(best_1) for j in s if i in best_2[j]]
        #print("# Candidates : {}".format(len(good)))
        # Slower and unclear how the crosscheck is performed
        # matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        # good = [((m.queryIdx, m.trainIdx)) for m in matcher.match(des1, des2)]

    src_pts = np.float32([kp1[m] for m, _ in good])
    dst_pts = np.float32([kp2[m] for _, m in good])
    try:
        def is_model_valid(model, X, y):
            m = model.coef_
            # no vertical flipping
            if m[0, 0] < 0:
                return False
            m_abs = np.abs(m)
            # scale
            if np.abs(np.log((m_abs[0, 0]+0.001) / (m_abs[1, 1]+0.001))) > np.log(1.4):
                return False
            # small rotation/shearing
            s = (m_abs[0, 0] + m_abs[1, 1]) / 2
            if max(m_abs[1, 0], m_abs[0, 1]) > 0.2 * s:
                return False
            return True

        regressor = RANSACRegressor(residual_threshold=1.0, max_trials=ransac_max_trials, min_samples=6,
                                    is_model_valid=is_model_valid)
        regressor.fit(src_pts, dst_pts)
        mask = regressor.inlier_mask_
        num_matches = int(np.sum(mask))

        matchesMask = mask.tolist()
        m1 = np.min(src_pts[mask], axis=0)
        m2 = np.max(src_pts[mask], axis=0)
        box1 = ((m1[0]-0.5)/h1, (m1[1]-0.5)/w1, (m2[0]-m1[0]+1)/h1, (m2[1]-m1[1]+1)/w1)
        m1 = np.min(dst_pts[mask], axis=0)
        m2 = np.max(dst_pts[mask], axis=0)
        box2 = ((m1[0]-0.5)/h2, (m1[1]-0.5)/w2, (m2[0]-m1[0]+1)/h2, (m2[1]-m1[1]+1)/w2)
    except ValueError:
        matchesMask = [True] * len(good)
        num_matches = 0
        regressor = None
        box1 = [0.0, 0.0, 1.0, 1.0]
        box2 = box1

    return num_matches, regressor, (src_pts, dst_pts), matchesMask, (box1, box2)