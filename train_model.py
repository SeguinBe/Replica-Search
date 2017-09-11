import argparse

import numpy as np
import tensorflow as tf
from replica_learn import input, export, model, dataset, evaluation, utils
from replica_search.index import IntegralImagesIndex
from tqdm import tqdm
import os
from random import shuffle
from functools import partial


class TrainingMethod:
    SIAMESE = 0
    TRIPLET = 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training-directory", required=True, help="Directory to save training progress")
    ap.add_argument("-d", "--training-dataset", required=True,
                    help="Pickle file containing the training ConnectedDataset")
    ap.add_argument("-b", "--validation-benchmark", required=True, help="Benchmark to compute the validation results")
    ap.add_argument("--epoch-init", default=0, type=int, help="Bypass the first epochs")
    args = vars(ap.parse_args())

    TRAINING_DIR = args['training_directory']
    EXPORT_DIR = os.path.join(TRAINING_DIR, 'export')

    training_dataset = utils.read_pickle(args['training_dataset'])
    assert isinstance(training_dataset, dataset.ConnectedDataset)

    validation_benchmark = utils.read_pickle(args['validation_benchmark'])
    assert isinstance(validation_benchmark, evaluation.Benchmark)

    PRETRAINED_DIR = '/home/seguin'
    EPOCH_INIT = args['epoch_init']
    NB_EPOCHS = 20
    TRAINING_METHOD = TrainingMethod.TRIPLET

    DATA_AUGMENTATION = True
    INCLUDE_LR_FRIP = True
    IMAGE_MAX_SIZE, IMAGE_INCREMENT = 320, 32

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = '0'
    # session_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=50)
    params = {
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'resnet_v1_50.ckpt'),
        'base_model': 'resnet50',
        'reducing_op': 'max',
        'contrastive_loss_margin': 0.6,
        'triplet_loss_margin': 0.001,
        'learning_rate': 0.00003
    }
    if True:
        params['base_model'] = 'vgg16'
        params['learning_rate'] = 0.00001
        params['triplet_loss_margin'] = 0.01
        params['pretrained_file'] = os.path.join(PRETRAINED_DIR, 'vgg_16.ckpt')

    estimator = tf.estimator.Estimator(model.model_fn,
                                       params=params,
                                       model_dir=TRAINING_DIR,
                                       config=estimator_config)

    training_preprocess = input.decode_and_resize(IMAGE_MAX_SIZE, IMAGE_INCREMENT,
                                                  input.data_augmentation_fn(INCLUDE_LR_FRIP) if DATA_AUGMENTATION else None)
    validation_preprocess = input.decode_and_resize(IMAGE_MAX_SIZE, IMAGE_INCREMENT)
    input_fn = partial(input.input_pairs_from_csv if TRAINING_METHOD == TrainingMethod.SIAMESE
                       else input.input_triplets_from_csv,
                       img_preprocessing_fn=training_preprocess,
                       num_epochs=1)
    # Perform one step just to export the model
    estimator.train(input_fn("/dev/null"), max_steps=1)

    FINAL_EPOCH = EPOCH_INIT+NB_EPOCHS
    for epoch in tqdm(range(EPOCH_INIT, FINAL_EPOCH+1)):
        # Export model
        model_export_dir = export.export_estimator(estimator, EXPORT_DIR, preprocess_function=validation_preprocess)

        # Compute search index
        index_filename = os.path.join(TRAINING_DIR, "index_{}.hdf5".format(epoch))
        if False or not os.path.exists(index_filename):
            # Export model
            loaded_model = export.StreamingModel(session_config, model_export_dir)
            with loaded_model:
                IntegralImagesIndex.build(loaded_model.output_generator_from_iterable(training_dataset.path_dict.items()),
                                          index_filename, save_feature_maps=False)
        else:
            print("Reusing {}".format(index_filename))
        last_index = IntegralImagesIndex(index_filename)
        search_function = last_index.search_one

        # Compute validation score
        benchmark_results = validation_benchmark.generate_evaluation_results(search_function)
        utils.write_pickle(benchmark_results, os.path.join(TRAINING_DIR, "benchmark_results_{}.pkl".format(epoch)))
        utils.write_as_summaries(TRAINING_DIR, {
            'eval/mean_ap': benchmark_results.mean_ap(),
            'eval/recall_at_20': benchmark_results.recall_at_n(20),
            'eval/recall_at_50': benchmark_results.recall_at_n(50),
            'eval/recall_at_100': benchmark_results.recall_at_n(100)
        })

        if epoch == FINAL_EPOCH:
            break

        # Generate training data
        training_file = os.path.join(TRAINING_DIR, "training_{}.csv".format(epoch))
        if TRAINING_METHOD == TrainingMethod.SIAMESE:
            training_examples = dataset.PairGenerator(3, 0.7).generate_training_pairs(training_dataset, search_function)
            shuffle(training_examples)
            # Very important to avoid NaN in the training
            training_examples = [p for p in training_examples
                                 if not np.allclose(last_index._get_feature(p[0]), last_index._get_feature(p[1]))]
        elif TRAINING_METHOD == TrainingMethod.TRIPLET:
            training_examples = training_dataset.sample_triplets(search_function, 10000)
            shuffle(training_examples)
            # Very important to avoid NaN in the training
            training_examples = [p for p in training_examples
                                 if not np.allclose(last_index._get_feature(p[0]), last_index._get_feature(p[1])) and
                                 not np.allclose(last_index._get_feature(p[0]), last_index._get_feature(p[2]))]
        else:
            raise NotImplementedError
        print("Generated {} training examples".format(len(training_examples)))
        training_dataset.save_examples_to_csv(training_file, training_examples)

        # Train
        estimator.train(input_fn(training_file))
