from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import List, Tuple

import numba
import numpy as np
import sklearn.svm as svm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from h5py import File
from time import time
from threading import Lock
from tqdm import tqdm, tqdm_notebook
import nmslib
import cv2
from functools import partial, lru_cache
import pickle

from .compression import compress_sparse_data, decompress_sparse_data
from .utils import Timer
from .match import match_feature_maps

from replica_learn.export import LoadedModel
import tensorflow as tf

"""@numba.jit(nogil=True)
def make_integral_image(feat_map):
    result = np.zeros((feat_map.shape[0]+1, feat_map.shape[1] + 1, feat_map.shape[2]), dtype=np.float64)
    result[1:, 1:, :] = np.cumsum(np.cumsum(feat_map, axis=1), axis=0)
    return result"""


ALPHA = 1.
SPATIAL_POOLING = True


@numba.jit(nopython=True, nogil=True)
def make_integral_image(feat_map):
    result = np.empty((feat_map.shape[0] + 1, feat_map.shape[1] + 1, feat_map.shape[2]), dtype=np.float64)
    if ALPHA != 1.:
        tmp = np.power(feat_map, ALPHA)
    else:
        tmp = feat_map.astype(np.float64)
    result[0, :, :] = 0
    result[:, 0, :] = 0
    for i in range(feat_map.shape[0]):
        line_sum = np.zeros((feat_map.shape[2]), dtype=np.float64)
        for j in range(feat_map.shape[1]):
            line_sum += tmp[i, j]
            result[i + 1, j + 1, :] = result[i, j + 1, :] + line_sum
    return result


@numba.jit(nopython=True, nogil=True)
def get_integral_from_integral_image(integral_image, y1, x1, y2, x2, result):
    result[:] = integral_image[y2, x2] + \
                integral_image[y1, x1] - \
                integral_image[y1, x2] - \
                integral_image[y2, x1]
    if ALPHA != 1.:
        result[:] = np.power(result + 0.1, 1 / ALPHA)


@numba.jit(nopython=True, nogil=True)
def get_score(integral_image, query_descriptor, y1, x1, y2, x2):
    tmp_descriptor = np.empty_like(query_descriptor)
    get_descriptor_from_integral_image(integral_image, y1, x1, y2, x2, tmp_descriptor)
    return np.sum(query_descriptor * tmp_descriptor)


@numba.jit(nopython=True, nogil=True)
def get_descriptor_from_integral_image(integral_image, y1, x1, y2, x2, result):
    if SPATIAL_POOLING:
        d = result.size // 4
        y_center, x_center = int(y1 + np.round((y2 - y1) / 2)), int(x1 + np.round((x2 - x1) / 2))
        get_integral_from_integral_image(integral_image, y1, x1, y_center, x_center, result[:d])
        get_integral_from_integral_image(integral_image, y_center, x1, y2, x_center, result[d:2 * d])
        get_integral_from_integral_image(integral_image, y1, x_center, y_center, x2, result[2 * d:3 * d])
        get_integral_from_integral_image(integral_image, y_center, x_center, y2, x2, result[3 * d:])
    else:
        get_integral_from_integral_image(integral_image, y1, x1, y2, x2, result)
    result[:] = result[:] / np.sqrt(np.sum(result * result) + 0.01)


@numba.jit(nopython=True, nogil=True)
def search_one_integral_image(integral_image, query_descriptor, query_ratio, step=1, min_size=2):
    best_y1, best_y2, best_x1, best_x2, best_score = 0, 0, 0, 0, -np.inf
    for y2 in range(0, integral_image.shape[0], step):
        for x2 in range(0, integral_image.shape[1], step):
            for y1 in range(0, y2, step):
                for x1 in range(0, x2, step):
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        continue
                    if 0.8 < (y2 - y1) / (x2 - x1) / query_ratio < 1.2:
                        score_tmp = get_score(integral_image, query_descriptor, y1, x1, y2, x2)
                        if score_tmp > best_score:
                            best_score = score_tmp
                            best_y1, best_y2, best_x1, best_x2 = y1, x1, y2, x2
    return best_score, (best_y1, best_y2, best_x1, best_x2)


# @numba.jit(nopython=True, nogil=True)
def search_refine(integral_image, query_descriptor, best_window, rf_steps, rf_iter):
    y1, x1, y2, x2 = best_window[0], best_window[1], best_window[2], best_window[3]
    h, w = integral_image.shape[0], integral_image.shape[1]
    best_score = get_score(integral_image, query_descriptor, y1, x1, y2, x2)

    for step in range(rf_steps, 0, -1):
        change = True
        c = 0
        while change & (c < rf_iter):
            change = False

            if y1 >= step:
                score_tmp = get_score(integral_image, query_descriptor, y1 - step, x1, y2, x2)
                if score_tmp > best_score:
                    y1 -= step
                    best_score = score_tmp
                    change = True

            if y1 < y2 - step:
                score_tmp = get_score(integral_image, query_descriptor, y1 + step, x1, y2, x2)
                if score_tmp > best_score:
                    y1 += step
                    best_score = score_tmp
                    change = True

            if y2 > y1 + step:
                score_tmp = get_score(integral_image, query_descriptor, y1, x1, y2 - step, x2)
                if score_tmp > best_score:
                    y2 -= step
                    best_score = score_tmp
                    change = True

            if y2 < h - step:
                score_tmp = get_score(integral_image, query_descriptor, y1, x1, y2 + step, x2)
                if score_tmp > best_score:
                    y2 += step
                    best_score = score_tmp
                    change = True

            if x1 >= step:
                score_tmp = get_score(integral_image, query_descriptor, y1, x1 - step, y2, x2)
                if score_tmp > best_score:
                    x1 -= step
                    best_score = score_tmp
                    change = True

            if x1 < x2 - step:
                score_tmp = get_score(integral_image, query_descriptor, y1, x1 + step, y2, x2)
                if score_tmp > best_score:
                    x1 += step
                    best_score = score_tmp
                    change = True

            if x2 > x1 + step:
                score_tmp = get_score(integral_image, query_descriptor, y1, x1, y2, x2 - step)
                if score_tmp > best_score:
                    x2 -= step
                    best_score = score_tmp
                    change = True

            if x2 < w - step:
                score_tmp = get_score(integral_image, query_descriptor, y1, x1, y2, x2 + step)
                if score_tmp > best_score:
                    x2 += step
                    best_score = score_tmp
                    change = True
            c += 1
    return best_score, (y1, x1, y2, x2)


def search_one_class_svm(search_features: np.ndarray, features: np.ndarray) -> np.ndarray:
    model = svm.OneClassSVM(kernel='rbf', gamma=1)
    model.fit(search_features)
    scores = model.decision_function(features)
    scores = scores.ravel()
    return scores


def search_svm(search_features: np.ndarray, negative_features: np.ndarray, features: np.ndarray) -> np.ndarray:
    y = np.zeros((len(search_features) + len(negative_features)))
    y[range(len(search_features))] = 1
    training_features = np.concatenate((search_features, negative_features), axis=0)
    model = svm.SVC(kernel='rbf', gamma=2)  # , class_weight={0: 1*len(search_inds)/float(len(negative_inds)),1: 1})
    model.fit(training_features, y)
    knowledge_model = svm.OneClassSVM()
    knowledge_model.fit(training_features)

    scores = model.decision_function(features)  # + knowledge_model.decision_function(features)
    scores = scores.ravel()
    return scores


class IntegralImagesIndex:
    class IndexType:
        BASE = 'base_index'
        HALF_DIM_PCA = 'half_dim_pca'

    def __init__(self, data_filename: str, base_index_key=IndexType.BASE, build_nn=False):
        print('Reading {}'.format(data_filename))
        self.data_filename = data_filename
        data_file = File(data_filename, mode='r')
        self.saved_model_file = data_file.attrs.get('saved_model_file', None)

        print("Using base index : {}".format(base_index_key))
        self.base_index_features = data_file[base_index_key]['features'].value
        self.base_index_inds_to_uids = data_file[base_index_key]['uids'].value.astype(str)
        self.base_index_uids_to_inds = {uid: i for i, uid in enumerate(self.base_index_inds_to_uids)}

        # if self.pca_components is not None or self.whiten:
        #    print("Nb components in PCA: {}".format(self.preprocessing.named_steps['pca'].components_.shape[0]))

        # Constructing the integral images
        if 'feature_maps' in data_file.keys():
            self.feature_maps = data_file['feature_maps']
        else:
            self.feature_maps = None

        if build_nn:
            self.index_nn = nmslib.init(method='seq_search', space='cosinesimil')
            self.index_nn.addDataPointBatch(self.base_index_features)
            self.index_nn.createIndex({}, print_progress=True)
        else:
            self.index_nn = None

        if self.saved_model_file is not None:
            if 'preprocessing' in data_file[base_index_key].keys():
                self.preprocessing = pickle.loads(bytes(data_file[base_index_key]['preprocessing'].value))
                assert isinstance(self.preprocessing, Pipeline)
            else:
                self.preprocessing = None
            self.loaded_model = LoadedModel(tf.ConfigProto(device_count={'GPU': 0}),
                                            self.saved_model_file,
                                            input_mode="image")
            self.loaded_model.__enter__()
        else:
            self.loaded_model = None

    def __repr__(self):
        return 'Index {} images, {}-d vectors, {} feature-maps'.format(len(self.base_index_inds_to_uids),
                                                                       self.base_index_features.shape[1],
                                                                       'with' if self.feature_maps else 'without')

    def get_number_of_images(self) -> int:
        return len(self.base_index_uids_to_inds)

    @classmethod
    def build(cls, feature_generator, data_filename: str, save_feature_maps=False, append=False, saved_model_file=None):
        """

        :param feature_generator: a generator outputting dict(output=<visual_f>, feature_map=<feature_map>)
        :param data_filename:
        :return:
        """
        with File(data_filename, mode='a' if append else 'x') as data_file:
            if saved_model_file is not None:
                data_file.attrs['saved_model_file'] = saved_model_file
            if save_feature_maps:
                feat_maps_group = data_file.require_group('feature_maps')
            else:
                feat_maps_group = None
            l_uids, l_features = [], []
            for output in feature_generator:
                uid = output['uid']
                l_uids.append(uid)
                l_features.append(output['output'])
                if save_feature_maps:
                    feat_maps_group.create_dataset(uid, data=np.void(compress_sparse_data(output['feature_map'])))
            base_index = data_file.require_group(cls.IndexType.BASE)
            if len(l_features) > 0:
                base_features = np.stack(l_features)
                base_uids = np.stack(l_uids)
                if append:
                    # TODO Merge properly in case adding already existing uids
                    base_features = np.concatenate([base_index['features'].value, base_features])
                    del base_index['features']
                    base_uids = np.concatenate([base_index['uids'].value, base_uids])
                    del base_index['uids']
                base_index.create_dataset('features', data=base_features)
                base_index.create_dataset('uids', data=base_uids)
            else:
                print('No elements added to index')

    @classmethod
    def add_transformed_index(cls, data_filename: str):
        with File(data_filename, mode='a') as data_file:
            # Load data
            base_index = data_file[IntegralImagesIndex.IndexType.BASE]
            base_features = base_index['features'].value
            base_uids = base_index['uids'].value
            # make PCA version
            preprocessing_steps = list()
            preprocessing_steps.append(('pre_normalize', Normalizer(norm='l2')))
            preprocessing_steps.append(("pca", PCA(n_components=base_features.shape[1] // 2)))
            preprocessing_steps.append(('post_normalize', Normalizer(norm='l2')))
            preprocessing_pipeline = Pipeline(preprocessing_steps)
            # Transform
            transformed_features = preprocessing_pipeline.fit_transform(base_features)
            # Save
            transformed_index = data_file.require_group(cls.IndexType.HALF_DIM_PCA)
            if 'features' in transformed_index:
                del transformed_index['features']
            transformed_index.create_dataset('features', data=transformed_features)
            if 'uids' in transformed_index:
                del transformed_index['uids']
            transformed_index.create_dataset('uids', data=base_uids)
            if 'preprocessing' in transformed_index:
                del transformed_index['preprocessing']
            transformed_index.create_dataset('preprocessing', data=np.void(pickle.dumps(preprocessing_pipeline)))

    def search(self, positive_ids: List[str], negative_ids: List[str],
               nb_results: int, filtered_ids=None) -> List[Tuple[str, float]]:
        # This part raises KeyError is elements not in the index
        # print(positive_ids)

        positive_features = np.stack([self.get_feature_from_uuid(uid) for uid in positive_ids])
        if len(negative_ids) > 0:
            negative_features = np.stack([self.get_feature_from_uuid(uid) for uid in negative_ids])
        else:
            negative_features = np.zeros((0, positive_features.shape[1]), np.float32)

        return self.search_from_features(positive_features, negative_features, nb_results, filtered_ids)

    def search_from_features(self, positive_features: np.ndarray, negative_features: np.ndarray, nb_results: int, filtered_ids=None):
        if filtered_ids is not None and len(filtered_ids) > 0:
            filtered_inds = [self.base_index_uids_to_inds[_id] for _id in set(filtered_ids)
                             if _id in self.base_index_uids_to_inds]
            features = self.base_index_features[filtered_inds]
            uid_list = self.base_index_inds_to_uids[filtered_inds]
        else:
            features = self.base_index_features
            uid_list = self.base_index_inds_to_uids

        if len(negative_features) > 0:
            scores = search_svm(positive_features, negative_features, features)
        elif len(positive_features) == 1:
            scores = (features @ positive_features[0])
        else:
            scores = search_one_class_svm(positive_features, features)

        results_ind = np.argsort(scores)[-1:-(min(nb_results, len(scores)) + 1):-1]
        return list(zip(uid_list[results_ind], scores[results_ind].astype(np.float64)))

    def search_from_image(self, image_bytes: bytes, *args, **kwargs):
        feature_vector = self.get_feature_from_image(image_bytes)
        return self.search_from_features(feature_vector[None],
                                         np.zeros((0, feature_vector.shape[0]), dtype=np.float32),
                                         *args, **kwargs)

    def search_one(self, positive_id: str, nb_results: int) -> List[Tuple[str, float]]:
        if self.index_nn is not None:
            results = self.index_nn.knnQuery(self.get_feature_from_uuid(positive_id), nb_results)
            return [(self.base_index_inds_to_uids[r[0]], -r[1]) for r in zip(*results)]
        else:
            return self.search([positive_id], [], nb_results)

    @lru_cache(maxsize=30000)
    def get_compressed_feature_map(self, uuid) -> bytes:
        assert self.feature_maps is not None, "Index does not contain feature maps"
        return bytes(self.feature_maps[uuid].value)

    def get_feature_map(self, uuid):
        assert self.feature_maps is not None, "Index does not contain feature maps"
        return decompress_sparse_data(self.get_compressed_feature_map(uuid))

    def _get_integral_image(self, uuid: str):
        assert self.feature_maps is not None, "Index does not contain feature maps"
        return make_integral_image(decompress_sparse_data(bytes(self.feature_maps[uuid].value)))

    def get_feature_from_uuid(self, uuid: str) -> np.ndarray:
        """

        :param uuid: key of the feature to be retrieved
        :return: a 1-D array of the corresponding feature
        """
        return self.base_index_features[self.base_index_uids_to_inds[uuid]]

    def get_feature_from_image(self, image_bytes: bytes) -> np.ndarray:
        """

        :param image_bytes: JPEG binary encoded image file
        :return: a 1-D array of the corresponding feature
        """
        assert self.loaded_model is not None
        feature_vector = self.loaded_model.predict(image_bytes)
        if self.preprocessing is not None:
            feature_vector = self.preprocessing.transform(feature_vector[None])[0]
        return feature_vector.astype(np.float32)

    def search_region(self, positive_id: str, region: np.ndarray, nb_results: int,
                      rerank_N=1000, filtered_ids=None) -> List:
        # Shortlist the valid elements
        time_begin = time()
        retrieved_ids = [r[0] for r in self.search([positive_id], [], rerank_N, filtered_ids)]
        print("Finding candidates {}s".format(time() - time_begin))
        # Find the query descriptor
        query_integral_image = self._get_integral_image(positive_id)
        query_descriptor = np.zeros((query_integral_image.shape[2] * (4 if SPATIAL_POOLING else 1),))
        h, w = query_integral_image.shape[0] - 1, query_integral_image.shape[1] - 1
        position = region.copy()
        position[2:] += position[:2]
        assert np.all(position <= 1) and np.all(position >= 0), 'Invalid region : {}'.format(position)
        position[[0, 2]] *= h
        position[[1, 3]] *= w
        get_descriptor_from_integral_image(query_integral_image, *np.round(position).astype(np.int),
                                           result=query_descriptor)
        query_ratio = np.round(region[2] * h) / np.round(region[3] * w)

        # Compute results
        scores = np.zeros((len(retrieved_ids),))
        scores[:] = -np.inf
        windows = np.zeros((len(retrieved_ids), 4), dtype=np.float)

        # @numba.jit(nopython=True, nogil=True)
        def _fn(i, f_map):
            # f_map = decompress_sparse_data(data)
            integral_image = make_integral_image(f_map)
            best_score, best_position = search_one_integral_image(integral_image, query_descriptor, query_ratio, step=1)
            # best_score, best_position = search_refine(integral_image, query_descriptor, best_position, 1, 30)
            # Refine position
            scores[i] = best_score
            h, w = integral_image.shape[0] - 1, integral_image.shape[1] - 1
            windows[i, :] = best_position
            windows[i, 0] /= h
            windows[i, 1] /= w
            windows[i, 2] /= h
            windows[i, 3] /= w
            windows[i, 2:] = windows[i, 2:] - windows[i, :2]

        time_begin = time()
        f_map_list = [bytes(self.feature_maps[uuid].value) for uuid in retrieved_ids]
        print("Load data {}s".format(time() - time_begin))
        time_begin = time()
        f_map_list = [decompress_sparse_data(f_map) for f_map in f_map_list]
        print("Decompress data {}s".format(time() - time_begin))

        time_begin = time()
        with ThreadPoolExecutor(max_workers=12) as e:
            e.map(_fn, range(len(f_map_list)), f_map_list, chunksize=20)
            # for i, f_map in enumerate(f_map_list):
            #    e.submit(_fn, i, f_map)
        print("MakeIntegralImage/Search {}".format(time() - time_begin))

        # Output the results
        results_ind = np.argsort(scores)[-1:-(min(nb_results, len(scores)) + 1):-1]
        return [(uuid, s, win_target.tolist())
                for (uuid, s, win_target) in zip(np.array(retrieved_ids)[results_ind],
                                                 scores[results_ind],
                                                 windows[results_ind])]

    @staticmethod
    def find_closest_pairs(features, max_threshold=0.065, min_threshold=-1, knn_limit=100,
                           method='seq_search', method_params=None):
        index_nn = nmslib.init(method=method, space='cosinesimil')
        index_nn.addDataPointBatch(features)
        print("Creating ANN-index")
        index_nn.createIndex(method_params, print_progress=True)
        print("Done")
        results = []
        batch_size = 1000
        for i in tqdm(range(0, len(features), batch_size)):
            inds = range(i, min(i + batch_size, len(features)))
            r = index_nn.knnQueryBatch(features[inds], k=knn_limit)
            for ind, (ids, dist) in zip(inds, r):
                for id in np.where(np.logical_and(dist < max_threshold, dist > min_threshold))[0]:
                    if ids[id] <= ind:
                        continue
                    results.append((ind, ids[id], dist[id]))
        return results

    def find_duplicates(self, max_threshold=0.065, min_threshold=-1, method='seq_search', method_params=None) -> List[
        Tuple[str, str, float]]:
        """
        :param max_threshold:
        :param min_threshold:
        :param method: 'seq_search', 'hnsw' etc...
        :param method_params: dict of params (for instance {'post': 2} for hnsw)
        :return:
        """
        results = self.find_closest_pairs(self.base_index_features, max_threshold, min_threshold, method=method, method_params=method_params)

        return sorted([(self.base_index_inds_to_uids[r[0]], self.base_index_inds_to_uids[r[1]], r[2]) for r in results],
                      key=lambda r: r[2])

    def find_clusters(self, max_threshold=0.04, min_samples=3, n_jobs=10) -> List[List[str]]:
        cluster_alg = DBSCAN(eps=max_threshold, min_samples=min_samples, n_jobs=n_jobs, metric='cosine',
                             algorithm='brute')
        clusters = cluster_alg.fit_predict(self.base_index_features)
        results = []
        for i in range(np.max(clusters) + 1):
            results.append(self.base_index_inds_to_uids[clusters == i])
        return results

    def make_distance_matrix(self, uids):
        is_present = np.array([uid in self.base_index_uids_to_inds.keys() for uid in uids], dtype=np.bool)

        features = np.stack([self.get_feature_from_uuid(uid) for uid in np.array(uids)[is_present]])
        distances_present = pairwise_distances(features, metric='euclidean')
        if np.all(is_present):
            return distances_present
        else:
            distances = np.ones([len(is_present), len(is_present)], dtype=distances_present.dtype)
            distances[np.outer(~is_present, is_present)] = 10
            distances[np.outer(is_present, ~is_present)] = 10
            distances[np.outer(~is_present, ~is_present)] = 0.01
            distances[np.outer(is_present, is_present)] = distances_present.ravel()
            return distances

    def match(self, uid1, uid2, dataset=None, **kwargs):
        f_map_1 = self.get_feature_map(uid1)
        f_map_2 = self.get_feature_map(uid2)

        num_matches, regressor, (src_pts, dst_pts), matchesMask, (box1, box2) = match_feature_maps(f_map_1, f_map_2, **kwargs)

        if dataset is not None:
            img1 = dataset.get_img(uid1, max_dim=1024)
            img2 = dataset.get_img(uid2, max_dim=1024)
            h1, w1, _ = f_map_1.shape
            h2, w2, _ = f_map_2.shape

            if regressor is not None:
                m1 = np.min(src_pts[matchesMask], axis=0)
                m2 = np.max(src_pts[matchesMask], axis=0)
                pts = np.float32([[m1[0], m1[1]], [m1[0], m2[1]], [m2[0], m2[1]], [m2[0], m1[1]]])
                dst = regressor.predict(pts)

                img1 = cv2.polylines(img1, [np.int32(pts * np.array(img1.shape[:2]) / np.array([h1, w1]))[:, ::-1]],
                                     True, 255, 3, cv2.LINE_AA)
                img2 = cv2.polylines(img2, [np.int32(dst * np.array(img2.shape[:2]) / np.array([h2, w2]))[:, ::-1]],
                                     True, 255, 3, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            # print(kp1)
            cv_kp1 = [cv2.KeyPoint(x, y, 20) for y, x in src_pts * np.array(img1.shape[:2]) / np.array([h1, w1])]
            cv_kp2 = [cv2.KeyPoint(x, y, 20) for y, x in dst_pts * np.array(img2.shape[:2]) / np.array([h2, w2])]
            cv_matches = [cv2.DMatch(i, i, 0) for i in range(len(matchesMask))]
            img3 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, cv_matches,
                                   None, **draw_params)
            return num_matches, img3
        else:
            return num_matches

    @staticmethod
    def _cnn_match_map_fn(pair, **kwargs):
        with Timer("decompress", disable=True):
            f_map_1 = decompress_sparse_data(pair[0])
            f_map_2 = decompress_sparse_data(pair[1])
        with Timer("matching", disable=True):
            n_matches, _, _, _, boxes = match_feature_maps(f_map_1, f_map_2, **kwargs)
        return n_matches, boxes

    def many_cnn_matches(self, pairs, n_workers=12, print_progress=False, **kwargs):
        # Creates a generator reading the compressed version of the feature_maps
        with Timer('Reading feature_maps', disable=True):
            binary_gen = [(self.get_compressed_feature_map(uid1), self.get_compressed_feature_map(uid2))
                          for uid1, uid2 in pairs]

        with Timer('Reranking Computation', disable=True):
            results = []
            #with Pool(n_workers) as p:
            #    for simple_result in tqdm(p.imap(partial(IntegralImagesIndex._cnn_match_map_fn, **kwargs), binary_gen,
            #                                     chunksize=20), total=len(pairs),
            #                              disable=not print_progress):
            #        results.append(simple_result)

            with ThreadPoolExecutor(max_workers=n_workers) as e:
                results = e.map(partial(IntegralImagesIndex._cnn_match_map_fn, **kwargs), binary_gen)
        return results

    def search_with_cnn_reranking(self, uid, nb_results: int, rerank_N=1000, filtered_ids=None, candidates=None,
                                  match_params=None):
        if candidates is None:
            candidates = [c[0] for c in self.search([uid], [], rerank_N, filtered_ids)]
        results = self.many_cnn_matches([(uid, c) for c in candidates], n_workers=12,
                                        **(match_params if match_params is not None else dict()))
        scores = [(c, r[0], r[1][1]) for c, r in zip(candidates, results)]
        return sorted(scores, key=lambda s: s[1], reverse=True)[:nb_results]

    @staticmethod
    def make_composed_search_function(base_index: 'IntegralImagesIndex', feature_map_index: 'IntegralImagesIndex',
                                      rerank_n: int, match_params=None):
        def _fn(uid: str, nb_results: int):
            results = base_index.search_one(uid, max(nb_results, rerank_n))
            reranked_results = feature_map_index.search_with_cnn_reranking(uid, nb_results,
                                                                           candidates=[r[0] for r in results],
                                                                           match_params=match_params)
            return [t[:2] for t in reranked_results]
        return _fn
