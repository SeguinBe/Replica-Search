import cv2
import numpy as np
from typing import List
from tqdm import tqdm
from scipy.misc import imread
from multiprocessing import Pool
from h5py import File
import pickle
try:
    import faiss
except:
    pass


class KPFeature:
    def __init__(self, keypoints, descriptors: np.ndarray, root=False):
        self.keypoints_pos = cv2.KeyPoint_convert(keypoints)
        self.keypoints_size = np.array([point.size for point in keypoints])
        self.keypoints_angle = np.array([point.angle for point in keypoints])
        self.keypoints_response = np.array([point.response for point in keypoints])
        self.descriptors = np.sqrt(descriptors) if root else descriptors


_sift_detector = cv2.xfeatures2d.SIFT_create()
_surf_upright_detector = cv2.xfeatures2d.SURF_create(upright=True)
_surf_detector = cv2.xfeatures2d.SURF_create()


def compute_features(img: np.ndarray,
                     type='surf_upright',
                     max_dim=720):
    """
    Compute the SIFT features for the given
    :param img: input image as RGB
    :return: the computed features
    """
    img_input = img.astype(np.uint8, copy=False)
    if max(img.shape[0], img.shape[1]) > max_dim:
        if img.shape[0] < img.shape[1]:
            new_size = (max_dim, int(img.shape[0]/img.shape[1]*max_dim))
        else:
            new_size = (int(img.shape[1]/img.shape[0]*max_dim), max_dim)
        img_input = cv2.resize(img_input, new_size)
    if len(img.shape) == 3:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
    if type == 'surf':
        detector = _surf_detector
    elif type == 'surf_upright':
        detector = _surf_upright_detector
    elif type == 'sift':
        detector = _sift_detector
    else:
        raise NotImplementedError
    return KPFeature(*detector.detectAndCompute(img_input, None))


def _fn(filename):
    img = imread(filename)
    f = compute_features(img)
    return f.descriptors


def gather_descriptors_for_images(filenames: List[str], processes=1):
    all_data = []

    if processes > 1:
        with Pool(processes) as p:
            for simple_result in tqdm(p.imap(_fn, filenames, chunksize=3), total=len(filenames)):
                all_data.append(simple_result)
    else:
        for filename in tqdm(filenames, desc="Extracting descriptors"):
            all_data.append(_fn(filename))

    return np.concatenate(all_data)


class BoWFeature:
    def __init__(self, sift_feature: KPFeature, quantizer):
        self.keypoints_pos = sift_feature.keypoints_pos
        self.keypoints_size = sift_feature.keypoints_size
        self.keypoints_angle = sift_feature.keypoints_angle
        self.words_inds = quantizer(sift_feature.descriptors)


def _fn2(args):
    uid, filename = args
    img = imread(filename)
    #print(uid, filename, img.shape)
    #if min(img.shape[:2]) < 0.1*max(img.shape[:2]):
    #    print(uid, filename, img.shape)
    #    return uid, None
    try:
        return uid, compute_features(img)
    except Exception as e:
        print(uid, filename, img.shape)
        return uid, None


class BoWIndex:

    @classmethod
    def build(cls, uid_filenames, data_filename: str, centroids_filename: str, append=False):
        """

        :param feature_generator: a generator outputting dict(output=<visual_f>, feature_map=<feature_map>)
        :param data_filename:
        :return:
        """
        centroids_data = np.load(centroids_filename)
        # Configure search index
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, centroids_data.shape[1], flat_config)
        index.add(centroids_data)
        quantizer = lambda descriptors: index.search(descriptors, 1)[1][:, 0]

        with File(data_filename, mode='a' if append else 'x') as data_file:
            with Pool(12) as p:
                for uid, sift_feature in tqdm(p.imap(_fn2, uid_filenames, chunksize=3), total=len(uid_filenames)):
                    if sift_feature is not None:
                        bow_feature = BoWFeature(sift_feature, quantizer)
                        data_file.create_dataset(uid, data=np.void(pickle.dumps(bow_feature)))
