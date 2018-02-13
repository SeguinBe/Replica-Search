import tensorflow as tf
from replica_learn.export import StreamingModel, LoadedModel
from replica_search.index import IntegralImagesIndex
from replica_search.model import QueryIterator
import argparse
import app
from glob import glob
import os
from tqdm import tqdm
from PIL import Image
from h5py import File
import pandas as pd


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-directory", required=True, help="Directory where the exported model lies")
    ap.add_argument("-o", "--output-filename", required=True,
                    help="Name of the created index file")
    ap.add_argument("-i", "--input-directory", required=False, help="Directory with the images, if not given uses the default database")
    ap.add_argument("--csv-file", required=False, help="Csv file with the images")
    ap.add_argument('--feature-maps', dest='feature_maps', action='store_true', help="Save the feature maps")
    ap.add_argument('--no-feature-maps', dest='feature_maps', action='store_false', help="Do not save the feature maps")
    ap.add_argument('--append', action='store_true', help="Append to index")
    ap.set_defaults(feature_maps=False)
    args = vars(ap.parse_args())
    MODEL_DIR = args['model_directory']
    INDEX_FILENAME = args['output_filename']
    print(args)

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = '0'

    if args['append']:
        with File(INDEX_FILENAME, mode='r') as data_file:
            skip_uids = set(data_file[IntegralImagesIndex.IndexType.BASE]['uids'].value.astype(str))
    else:
        skip_uids = set()

    if args.get('input_directory') is None and args.get('csv_file') is None:
        def check_ratio_fn(l):
            p = l.get_image_path()
            img = Image.open(p)
            w, h = img.width, img.height
            if 0.11 < w/h < 9:
                return l.uid, p
            else:
                print('Wrong ratio for {}'.format(p))
                return l.uid, None
        def filter_fn(l):
            uid, p = l
            if p in [
                        '/mnt/project_replica/datasets/cini/3A/3A_284.jpg',  # Bad ratio
                        '/mnt/project_replica/datasets/cini/27A/27A_209.jpg',  # Corrupted
                        '/mnt/project_replica/datasets/cini/40B/40B_176.jpg',  # Corrupted
                        '/mnt/project_replica/datasets/cini/64B/64B_715.jpg',  # Corrupted
                        '/mnt/project_replica/datasets/cini/90B/90B_401.jpg',  # Corrupted
                     ]:
                return False
            elif uid in skip_uids:
                return False
            else:
                return True
        iterable = QueryIterator(app.Session().query(app.model.ImageLocation),
                                 fn=lambda l: (l.uid, l.get_image_path()))
        iterable = filter(filter_fn, iterable)
        iterable = list(iterable)
        print(iterable[:10])
    elif args.get('input_directory'):
        iterable = [(os.path.basename(f), f) for f in glob(args['input_directory'])]
    elif args.get('csv_file'):
        d = pd.read_csv(args['csv_file'])
        iterable = list(zip(d.uid, d.path))
    else:
        raise NotImplementedError

    loaded_model = StreamingModel(session_config, MODEL_DIR)
    with loaded_model:
        IntegralImagesIndex.build(loaded_model.output_generator_from_iterable(iterable),
                                  INDEX_FILENAME, save_feature_maps=args['feature_maps'], append=args['append'])

    print("Add secondary index")
    IntegralImagesIndex.add_transformed_index(INDEX_FILENAME)

    #loaded_model = LoadedModel(session_config, MODEL_DIR)
    #with loaded_model:
    #    for uid, filename in tqdm(iterable):
    #        loaded_model.predict(filename)

