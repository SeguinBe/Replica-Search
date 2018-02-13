import argparse

import numpy as np
import tensorflow as tf
from replica_learn import input, export, model, dataset, evaluation, utils
from replica_search.index import IntegralImagesIndex
from tqdm import tqdm
import os
from random import shuffle
from functools import partial
from sacred import Experiment
ex = Experiment()


class TrainingMethod:
    SIAMESE = 'SIAMESE'
    TRIPLET = 'TRIPLET'


PRETRAINED_DIR = '/mnt/cluster-nas/benoit/pretrained_nets'


@ex.config
def my_config():
    training_method = TrainingMethod.TRIPLET
    training_dir = '/scratch/benoit/tensorboard_matcher'
    training_dataset = '/scratch/benoit/dataset_wga.pkl'
    validation_benchmark = '/scratch/benoit/benchmark_wga.pkl'
    image_resizing = {
        'max_size': 320,
        'increment': 32
    }
    data_augmentation = {
        'activated': False,
        'include_flip_lr': True
    }
    model_params = {
        'base_model': 'vgg16',
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'vgg_16.ckpt'),
        'reducing_op': 'max',
        'blocks': 5,
        'contrastive_loss_margin': 0.1,
        'triplet_loss_margin': 0.01,
        'learning_rate': 0.000005,
        'decay_rate': 0.85,
        'decay_steps': 2000,
        'train_batch_norm': False,
        'matcher_params': {
            'nb_attention_layers': 0
        }
    }
    nb_epochs = 40

@ex.named_config
def local_dataset():
    training_dataset = '/home/seguin/dataset_wga.pkl'
    validation_benchmark = '/home/seguin/benchmark_wga.pkl'
    training_dir = '/home/seguin/tensorboard'

@ex.named_config
def resnet_50():
    model_params = {
        'base_model': 'resnet50',
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'resnet_v1_50.ckpt'),
        'triplet_loss_margin': 0.01,
        'learning_rate': 0.00001,
        'blocks': 4
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
def experiment(model_name, training_dataset, validation_benchmark, training_method,
               image_resizing, data_augmentation, model_params, nb_epochs, training_dir):

    TRAINING_DIR = '{}/{}'.format(training_dir, model_name)
    EXPORT_DIR = os.path.join(TRAINING_DIR, 'export')

    training_dataset = utils.read_pickle(training_dataset)
    assert isinstance(training_dataset, dataset.ConnectedDataset)

    validation_benchmark = utils.read_pickle(validation_benchmark)
    assert isinstance(validation_benchmark, evaluation.Benchmark)


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
                                                  input.data_augmentation_fn(data_augmentation['include_flip_lr'])
                                                  if data_augmentation['activated'] else None)
    validation_preprocess = input.decode_and_resize(image_resizing['max_size'], image_resizing['increment'])
    input_fn = partial(input.input_pairs_from_csv if training_method == TrainingMethod.SIAMESE
                       else input.input_triplets_from_csv,
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
        utils.write_pickle(benchmark_results, os.path.join(TRAINING_DIR, "benchmark_results_{}.pkl".format(epoch)))
        utils.write_as_summaries(TRAINING_DIR, {
            'eval/mean_ap': benchmark_results.mean_ap(),
            'eval/recall_at_20': benchmark_results.recall_at_n(20),
            'eval/recall_at_50': benchmark_results.recall_at_n(50),
            'eval/recall_at_100': benchmark_results.recall_at_n(100)
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
