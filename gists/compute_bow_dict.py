"""
Do not forget for faiss

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/soliveir/anaconda3/lib
export LD_PRELOAD=/home/soliveir/anaconda3/lib/libmkl_core.so:/home/soliveir/anaconda3/lib/libmkl_sequential.so

"""

import numpy as np
import os
import argparse
from replica_search.model import ImageLocation
from replica_search import bow
import app
import faiss


def compute_and_save_descriptors(nb_images, output_file, nb_processes):
    filenames = [i.get_image_path() for i in app.Session().query(ImageLocation)]
    filenames_sample = np.random.choice(filenames, nb_images, replace=False)
    descriptors = bow.gather_descriptors_for_images(filenames_sample, nb_processes)
    np.save(output_file, descriptors)


def compute_kmeans(nb_clusters, input_file, output_file):
    data = np.load(input_file).astype(np.float32, copy=False)
    print(data.shape)
    d = data.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    kmeans = faiss.Clustering(d, nb_clusters)
    kmeans.train(data, index)
    centroids = faiss.vector_float_to_array(kmeans.centroids).reshape([-1, d])
    np.save(output_file, centroids)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--nb-images", default=20000, help="Nb of images for extracting descriptors")
    ap.add_argument("-d", "--descriptor-file", required=True, help="File with the extracted descriptors (.npy)")
    ap.add_argument("-o", "--cluster-file", required=False,
                    help='File with the computed cluster centers (.npy)')
    ap.add_argument("-c", "--nb-clusters", required=False, type=int,
                    help='Nb of clusters (.npy)')
    args = ap.parse_args()

    if not os.path.exists(args.descriptor_file):
        compute_and_save_descriptors(args.nb_images, args.descriptor_file, 10)
    else:
        print("Skipping descriptor computation, reusing {}".format(args.descriptor_file))

    if args.cluster_file is not None:
        compute_kmeans(args.nb_clusters, args.descriptor_file, args.cluster_file)
