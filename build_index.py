import tensorflow as tf
from replica_learn.export import StreamingModel, LoadedModel
from replica_search.index import IntegralImagesIndex
from replica_search.model import QueryIterator
import argparse
import app
from glob import glob
import os
from tqdm import tqdm


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-directory", required=True, help="Directory where the exported model lies")
    ap.add_argument("-o", "--output-filename", required=True,
                    help="Name of the created index file")
    ap.add_argument("-i", "--input-directory", required=False, help="Directory with the images, if not given uses the default database")
    ap.add_argument('--feature-maps', dest='feature_maps', action='store_true', help="Save the feature maps")
    ap.add_argument('--no-feature-maps', dest='feature_maps', action='store_false', help="Do not save the feature maps")
    ap.set_defaults(feature_maps=True)
    args = vars(ap.parse_args())
    MODEL_DIR = args['model_directory']
    INDEX_FILENAME = args['output_filename']
    print(args)

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = '0'

    if args.get('input_directory') is None:
        iterable = QueryIterator(app.Session().query(app.model.ImageLocation),
                                 fn=lambda l: (l.uid, l.get_image_path()))
    else:
        iterable = [(os.path.basename(f), f) for f in glob(args['input_directory'])]

    loaded_model = StreamingModel(session_config, MODEL_DIR)
    with loaded_model:
        IntegralImagesIndex.build(loaded_model.output_generator_from_iterable(iterable),
                                  INDEX_FILENAME, save_feature_maps=args['feature_maps'])

    print("Add secondary index")
    IntegralImagesIndex.add_transformed_index(INDEX_FILENAME)

    #loaded_model = LoadedModel(session_config, MODEL_DIR)
    #with loaded_model:
    #    for uid, filename in tqdm(iterable):
    #        loaded_model.predict(filename)

