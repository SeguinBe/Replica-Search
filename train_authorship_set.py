import os
from functools import partial
from random import shuffle

import tensorflow as tf
from tqdm import tqdm
import os
import json
import numpy as np

from train_authorship_set_configs import ex, ModelParams, TrainingParams
from replica_learn import input, export, model2, dataset
from replica_search.index import IntegralImagesIndex

try:
    import better_exceptions
except:
    print('/!\ W -- Not able to import package better_exceptions')
    pass


@ex.automain
def experiment(model_name, author_csv, training_csv, validation_csv, testing_csv,
               training_index_file, validation_index_file, model_params, training_params,
               training_dir, _config):
    TRAINING_DIR = '{}/{}'.format(training_dir, model_name)
    EXPORT_DIR = os.path.join(TRAINING_DIR, 'export')

    if not os.path.isdir(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)
    else:
        assert _config.get('restore_model'), \
            '{} already exists, you cannot use it as output directory. ' \
            'Set "restore_model=True" to continue training'.format(TRAINING_DIR)

    # Loading data
    training_dataset = dataset.ClassificationDataset(training_csv, author_csv)
    image_index = IntegralImagesIndex(training_index_file, build_nn=False)
    validation_dataset = dataset.ClassificationDataset(validation_csv, author_csv)
    validation_index = IntegralImagesIndex(validation_index_file, build_nn=False)

    model_params = ModelParams.from_dict(model_params)
    training_params = TrainingParams.from_dict(training_params)
    model_params.n_classes = validation_dataset.get_number_classes()
    if training_params.lr_decay_steps == 'auto':
        training_params.lr_decay_steps = len(training_dataset.path_dict)/training_params.batch_size

    # Save config to json
    with open(os.path.join(TRAINING_DIR, 'config.json'), 'w') as f:
        json.dump(_config, f, indent=4)

    if training_params.weighted_loss:
        raise NotImplementedError
        model_params['class_weights'] = training_dataset.get_weight_vector()
    print("Number of classes : {}".format(model_params.n_classes))

    # Estimator Parameters
    estimator_params = dict()
    estimator_params['image_embeddings'] = image_index.base_index_features
    estimator_params['class_ids'] = np.array([training_dataset.reverse_class_dict[uid]
                                             for uid in image_index.base_index_inds_to_uids])
    estimator_params['model_params'] = model_params
    estimator_params['training_params'] = training_params

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = '0'
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=50)

    # Create estimator
    estimator = tf.estimator.Estimator(model2.model_authorship_fn,
                                       params=estimator_params,
                                       model_dir=TRAINING_DIR,
                                       config=estimator_config)

    # Prepare training data
    class_uids = [training_dataset.class_dict[i] for i in range(training_dataset.get_number_classes())]
    training_input_fn = partial(input.input_set_classification_train,
                                batch_size=training_params.batch_size,
                                uid_to_id_dict=image_index.base_index_uids_to_inds,
                                class_uids=class_uids,
                                num_epochs=1)
    validation_input_fn = partial(input.input_set_classification_inference,
                                  num_epochs=1)
    # Perform one step just to export the model
    # estimator.train(training_input_fn([]))

    for epoch in tqdm(range(training_params.nb_epochs)):
        # Generate training data
        training_file = os.path.join(TRAINING_DIR, "training_{}.csv".format(epoch))
        training_examples = training_dataset.generate_training_samples(id_only=True)
        shuffle(training_examples)

        # pd.DataFrame(training_examples, columns=['filename', 'label']).to_csv(training_file)
        # print("Generated {} training examples".format(len(training_examples)))

        # Train
        estimator.train(training_input_fn(training_examples))

        # Compute validation score
        validation_ids, labels = zip(*validation_dataset.generate_training_samples(id_only=True))
        validation_elements = np.stack([validation_index.get_feature_from_uuid(uid) for uid in validation_ids])
        estimator.evaluate(validation_input_fn(validation_elements, labels=labels))

        estimator.export_savedmodel(EXPORT_DIR,
                                    tf.estimator.export.build_raw_serving_input_receiver_fn(
                                        {'inputs': tf.placeholder(tf.float32, [None,
                                                                               estimator_params['image_embeddings'].shape[1]])}))

        # export.export_estimator(estimator, EXPORT_DIR, preprocess_function=validation_preprocess)
