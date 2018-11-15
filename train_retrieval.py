import argparse

import numpy as np
import tensorflow as tf
from replica_learn import input, export, model, dataset, evaluation, utils
from replica_search.index import IntegralImagesIndex
from tqdm import tqdm
import os
# Tensorflow logging level
from logging import WARNING  # import  DEBUG, INFO, ERROR for more/less verbosity
tf.logging.set_verbosity(WARNING)
from random import shuffle
from functools import partial
from sacred import Experiment
import json
try:
    import better_exceptions
except ImportError:
    print('/!\ W -- Not able to import package better_exceptions')
    pass
ex = Experiment()


class TrainingMethod:
    SIAMESE = 'SIAMESE'
    TRIPLET = 'TRIPLET'


PRETRAINED_DIR = '/home/seguin/cluster-nas/benoit/pretrained_nets'


@ex.config
def my_config():
    training_method = TrainingMethod.TRIPLET
    training_dir = '/scratch/benoit/tensorboard_matcher'
    training_dataset = '/scratch/benoit/dataset_wga.pkl'
    validation_benchmark = '/scratch/benoit/benchmark_wga.pkl'
    testing_benchmark = None
    image_resizing = {
        'max_size': 320,
        'increment': 32
    }
    data_augmentation = {
        'activated': True,
        'params': {
            'lr_flip': True,
            'rotation': False,
            'zoom': True,
            'color': False
        }
    }
    model_params = {
        'batch_size': 8,
        'reducing_op': 'max',
        'contrastive_loss_margin': 0.1,
        'triplet_loss_margin': 0.01,
        'learning_rate': 0.000005,
        'decay_rate': 0.85,
        'decay_steps': 8000,
        'train_batch_norm': False,
        'matcher_params': {
            'nb_attention_layers': 0
        }
    }
    nb_epochs = 40

@ex.named_config
def local_dataset():
    training_dataset = '/home/seguin/experiment_data_wga/dataset_1_training.pkl'
    validation_benchmark = '/home/seguin/experiment_data_wga/benchmark_validation_1.pkl'
    testing_benchmark = '/home/seguin/experiment_data_wga/benchmark_2.pkl'
    training_dir = '/home/seguin/cluster-nas/wga_experiments'

@ex.named_config
def resnet_50():
    model_params = {
        'base_model': 'resnet50',
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'resnet_v1_50.ckpt'),
        'learning_rate': 0.00001,
        'blocks': 4,
        'weight_decay': 0.00002
    }

@ex.named_config
def vgg16():
    model_params = {
        'base_model': 'vgg16',
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'vgg_16.ckpt'),
        'blocks': 5,
        'learning_rate': 0.00001,
        'weight_decay': 0.0005
    }

@ex.named_config
def xception():
    model_params = {
        'base_model': 'xception',
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'xception.ckpt'),
        'triplet_loss_margin': 0.01,
        'learning_rate': 0.00001,
        'blocks': 15
    }

@ex.named_config
def xception_fused():
    model_params = {
        'base_model': 'xception_fused',
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'xception_fused.ckpt'),
        'triplet_loss_margin': 0.01,
        'learning_rate': 0.00001,
        'blocks': 15
    }

@ex.named_config
def user_experiment():
    training_dir = '/scratch/benoit/experiment_models'
    data_augmentation = {'activated': False}
    nb_epochs = 20

@ex.automain
def experiment(model_name, training_dataset, validation_benchmark, testing_benchmark, training_method,
               image_resizing, data_augmentation, model_params, nb_epochs, training_dir, _config):

    TRAINING_DIR = '{}/{}'.format(training_dir, model_name)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    EXPORT_DIR = os.path.join(TRAINING_DIR, 'export')

    # Save config
    with open(os.path.join(TRAINING_DIR, 'config.json'), 'w') as f:
        json.dump(_config, f, indent=4, sort_keys=True)

    training_dataset = utils.read_pickle(training_dataset)
    assert isinstance(training_dataset, dataset.ConnectedDataset)

    validation_benchmark = utils.read_pickle(validation_benchmark)
    assert isinstance(validation_benchmark, evaluation.Benchmark)

    if testing_benchmark is not None:
        testing_benchmark = utils.read_pickle(testing_benchmark)
        assert isinstance(testing_benchmark, evaluation.Benchmark)

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = '0'
    # session_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=50)

    estimator = tf.estimator.Estimator(model.model_fn,
                                       params=model_params,
                                       model_dir=TRAINING_DIR,
                                       config=estimator_config)

    training_preprocess = input.decode_and_resize(image_resizing['max_size'], image_resizing['increment'],
                                                  input.data_augmentation_fn(**data_augmentation['params'])
                                                  if data_augmentation['activated'] else None)
    validation_preprocess = input.decode_and_resize(image_resizing['max_size'], image_resizing['increment'])
    input_fn = partial(input.input_pairs_from_csv if training_method == TrainingMethod.SIAMESE
                       else input.input_triplets_from_csv,
                       batch_size=model_params['batch_size'],
                       img_preprocessing_fn=training_preprocess,
                       num_epochs=1)
    # Perform one step just to export the model
    estimator.train(input_fn("/dev/null"), max_steps=1)

    for epoch in tqdm(range(nb_epochs+1)):
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
        utils.write_pickle(benchmark_results, os.path.join(TRAINING_DIR, "benchmark_validation_{}.pkl".format(epoch)))
        utils.write_as_summaries(TRAINING_DIR, {
            'eval/mean_ap': benchmark_results.mean_ap(),
            'eval/recall_at_20': benchmark_results.recall_at_n(20),
            'eval/recall_at_50': benchmark_results.recall_at_n(50),
            'eval/recall_at_100': benchmark_results.recall_at_n(100)
        })
        if testing_benchmark is not None:
            benchmark_results = testing_benchmark.generate_evaluation_results(search_function)
            utils.write_pickle(benchmark_results, os.path.join(TRAINING_DIR, "benchmark_testing_{}.pkl".format(epoch)))
            utils.write_as_summaries(TRAINING_DIR, {
                'test/mean_ap': benchmark_results.mean_ap(),
                'test/recall_at_20': benchmark_results.recall_at_n(20),
                'test/recall_at_50': benchmark_results.recall_at_n(50),
                'test/recall_at_100': benchmark_results.recall_at_n(100)
            })

        if epoch == nb_epochs:
            break

        # Generate training data
        training_file = os.path.join(TRAINING_DIR, "training_{}.csv".format(epoch))
        if training_method == TrainingMethod.SIAMESE:
            training_examples = dataset.PairGenerator(3, 0.7).generate_training_pairs(training_dataset, search_function)
            shuffle(training_examples)
            # Very important to avoid NaN in the training
            training_examples = [p for p in training_examples
                                 if not np.allclose(last_index._get_feature(p[0]), last_index._get_feature(p[1]))]
        elif training_method == TrainingMethod.TRIPLET:
            training_examples = training_dataset.sample_triplets(search_function, 10000,
                                                                 margin=model_params['triplet_loss_margin'])
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
