import os
from functools import partial
from random import shuffle

import tensorflow as tf
from tqdm import tqdm
import os
import json

from train_authorship_configs import ex
from replica_learn import input, export, model, dataset


@ex.automain
def experiment(model_name, author_csv, training_csv, validation_csv, testing_csv,
               image_resizing, data_augmentation, model_params, nb_epochs, training_dir, _config):
    TRAINING_DIR = '{}/{}'.format(training_dir, model_name)
    EXPORT_DIR = os.path.join(TRAINING_DIR, 'export')

    training_dataset = dataset.ClassificationDataset(training_csv, author_csv)
    validation_dataset = dataset.ClassificationDataset(validation_csv, author_csv)
    model_params['n_classes'] = validation_dataset.get_number_classes()
    if model_params['decay_steps'] == 'auto':
        model_params['decay_steps'] = len(training_dataset.path_dict)/model_params['batch_size']

    # Save config to json
    os.makedirs(TRAINING_DIR, exist_ok=True)
    with open(os.path.join(TRAINING_DIR, 'config.json'), 'w') as f:
        json.dump(_config, f, indent=4)

    if model_params['weighted_loss']:
        model_params['class_weights'] = training_dataset.get_weight_vector()
    print("Number of classes : {}".format(model_params['n_classes']))

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = '0'
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=50)

    estimator = tf.estimator.Estimator(model.model_authorship_fn,
                                       params=model_params,
                                       model_dir=TRAINING_DIR,
                                       config=estimator_config)

    training_preprocess = input.decode_and_resize(image_resizing['max_size'], image_resizing['increment'],
                                                  input.data_augmentation_fn(
                                                      lr_flip=data_augmentation['include_flip_lr'],
                                                      rotation=data_augmentation['include_rotation'],
                                                      zoom=data_augmentation['include_zoom'],
                                                      color=data_augmentation['include_color'])
                                                  if data_augmentation['activated'] else None)
    validation_preprocess = input.decode_and_resize(image_resizing['max_size'], image_resizing['increment'])
    training_input_fn = partial(input.input_img_label_pairs,
                                batch_size=model_params['batch_size'],
                                img_preprocessing_fn=training_preprocess,
                                num_epochs=1)
    validation_input_fn = partial(input.input_img_label_pairs,
                                  img_preprocessing_fn=validation_preprocess,
                                  num_epochs=1)
    # Perform one step just to export the model
    # estimator.train(training_input_fn([]))

    for epoch in tqdm(range(nb_epochs)):
        # Generate training data
        training_file = os.path.join(TRAINING_DIR, "training_{}.csv".format(epoch))
        training_examples = training_dataset.generate_training_samples()
        shuffle(training_examples)
        # pd.DataFrame(training_examples, columns=['filename', 'label']).to_csv(training_file)
        # print("Generated {} training examples".format(len(training_examples)))

        # Train
        estimator.train(training_input_fn(training_examples))

        # Compute validation score
        estimator.evaluate(validation_input_fn(validation_dataset.generate_training_samples()))

    # export.export_estimator(estimator, EXPORT_DIR, preprocess_function=validation_preprocess)
