import numpy as np
import tensorflow as tf


def input_pairs_from_csv(csv_filename, img_preprocessing_fn, batch_size=8, num_threads=8, num_epochs=None):
    def input_fn():
        filename_queue = tf.train.string_input_producer([csv_filename], num_epochs=num_epochs)

        # Skip lines that have already been processed
        reader = tf.TextLineReader(name='CSV_Reader')
        key, value = reader.read(filename_queue, name='file_reading_op')

        # value = tf.Print(value, [value])

        default_line = [['None'], ['None'], [0.0]]
        img_path1, img_path2, label = tf.decode_csv(value, record_defaults=default_line, field_delim=',',
                                                    name='csv_reading_op')

        # Preprocess
        to_be_batched = dict()
        to_be_batched['images_1'], to_be_batched['image_sizes_1'] = img_preprocessing_fn(tf.read_file(img_path1))
        to_be_batched['images_2'], to_be_batched['image_sizes_2'] = img_preprocessing_fn(tf.read_file(img_path2))
        to_be_batched['labels'] = label[None]

        # Batch
        capacity = batch_size * num_threads * 3
        batch = tf.train.batch(to_be_batched,
                               batch_size=batch_size,
                               num_threads=num_threads,
                               capacity=capacity,
                               # min_after_dequeue=batch_size * 3,
                               allow_smaller_final_batch=True)

        return batch, batch['labels']

    return input_fn


def input_triplets_from_csv(csv_filename, img_preprocessing_fn, batch_size=8, num_threads=8, num_epochs=None):
    def input_fn():
        filename_queue = tf.train.string_input_producer([csv_filename], num_epochs=num_epochs)

        # Skip lines that have already been processed
        reader = tf.TextLineReader(name='CSV_Reader')
        key, value = reader.read(filename_queue, name='file_reading_op')

        default_line = [['None'], ['None'], ['None']]
        img_path1, img_path2, img_path3 = tf.decode_csv(value, record_defaults=default_line, field_delim=',',
                                                        name='csv_reading_op')

        # Preprocess
        to_be_batched = dict()
        to_be_batched['images_1'], to_be_batched['image_sizes_1'] = img_preprocessing_fn(tf.read_file(img_path1))
        to_be_batched['images_2'], to_be_batched['image_sizes_2'] = img_preprocessing_fn(tf.read_file(img_path2))
        to_be_batched['images_3'], to_be_batched['image_sizes_3'] = img_preprocessing_fn(tf.read_file(img_path3))

        # Batch
        capacity = batch_size * num_threads * 3
        batch = tf.train.batch(to_be_batched,
                               batch_size=batch_size,
                               num_threads=num_threads,
                               capacity=capacity,
                               # min_after_dequeue=batch_size * 3,
                               allow_smaller_final_batch=True)



        return batch, None

    return input_fn


def input_uid_filename_from_csv(csv_filename, img_preprocessing_fn, batch_size=16, num_threads=8, num_epochs=None):
    def input_fn():
        csv_filenames = [csv_filename] if csv_filename is not None else []
        filename_queue = tf.train.string_input_producer(csv_filenames, num_epochs=num_epochs)

        # Skip lines that have already been processed
        reader = tf.TextLineReader(name='CSV_Reader')
        key, value = reader.read(filename_queue, name='file_reading_op')

        default_line = [['None'], ['None']]
        img_uid, img_path = tf.decode_csv(value, record_defaults=default_line, field_delim=',',
                                          name='csv_reading_op')

        # Preprocess
        to_be_batched = dict()
        to_be_batched['images_1'], to_be_batched['image_sizes_1'] = img_preprocessing_fn(tf.read_file(img_path))
        to_be_batched['uids'] = img_uid

        # Batch
        capacity = batch_size * num_threads * 3
        batch = tf.train.batch(to_be_batched,
                               batch_size=batch_size,
                               num_threads=num_threads,
                               capacity=capacity,
                               allow_smaller_final_batch=True)

        return batch, None

    return input_fn


def decode_and_resize(max_size, increment, data_augmentation_fn=None):
    def fn(raw_input):
        with tf.variable_scope('Preprocess'):
            decoded_image = tf.cast(tf.image.decode_jpeg(raw_input, channels=3), tf.float32)
            if data_augmentation_fn:
                decoded_image = data_augmentation_fn(decoded_image)
            original_shape = tf.cast(tf.shape(decoded_image)[:2], tf.float32)
            ratio = tf.reduce_min(max_size / original_shape)
            new_shape = original_shape * ratio
            rounded_shape = tf.cast(tf.round(new_shape / increment) * increment, tf.int32)
            resized_image = tf.image.resize_images(decoded_image, rounded_shape)
            paddings = tf.minimum(rounded_shape - 1, max_size - rounded_shape)
            # Do as much reflecting padding as possible to avoid screwing the batch_norm statistics
            padded_image = tf.pad(resized_image, [[0, paddings[0]], [0, paddings[1]], [0, 0]],
                                  mode='REFLECT')
            padded_image = tf.pad(padded_image, [[0, max_size - rounded_shape[0] - paddings[0]],
                                                 [0, max_size - rounded_shape[1] - paddings[1]], [0, 0]],
                                  mode='CONSTANT')
            padded_image.set_shape([max_size, max_size, 3])
            return padded_image, rounded_shape

    return fn


def data_augmentation_fn(lr_flip=True):
    def fn(img_tensor):
        with tf.variable_scope('DataAugmentation'):
            if lr_flip:
                img_tensor = tf.image.random_flip_left_right(img_tensor)
            img_tensor = random_zoom(img_tensor, max_zoom=0.15)
            img_tensor = random_rotation(img_tensor)
            # img_tensor = tf.image.random_brightness(img_tensor, max_delta=10)  # Tends to saturate the image
            img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.)
            img_tensor = tf.image.random_hue(img_tensor, max_delta=0.15)
            img_tensor = tf.image.random_saturation(img_tensor, lower=0.8, upper=1.2)
            return img_tensor

    return fn


def random_rotation(img, max_rotation=0.1, crop=True):
    with tf.name_scope('RandomRotation'):
        rotation = tf.random_uniform([], -max_rotation, max_rotation)
        rotated_image = tf.contrib.image.rotate(img, rotation)
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            # see https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l * tf.cos(rotation) - old_s * tf.sin(rotation)) / tf.cos(2 * rotation)
            new_s = (old_s - tf.sin(rotation) * new_l) / tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.ceil((h - new_h) / 2), tf.int32), tf.cast(tf.ceil((w - new_w) / 2), tf.int32)
            rotated_image = rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :]
        return rotated_image


def random_zoom(input_image, max_zoom=0.15):
    with tf.name_scope('RandomZoom'):
        zoom = tf.random_uniform([], 0, max_zoom)
        input_size = tf.shape(input_image)[:2]
        new_size = tf.cast(tf.cast(input_size, tf.float32) * (1. - zoom), tf.int32)
        crop = tf.random_crop(input_image, tf.concat([new_size, [3]], 0))
        return tf.image.resize_images(crop, input_size)
