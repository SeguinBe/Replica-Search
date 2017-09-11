import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim, layers
from tensorflow.contrib.slim import nets


def global_pool_layer(feature_maps, shapes, pooling_method: str, normalize_output=True):
    reducing_op = {
        'max': tf.reduce_max,
        'sum': tf.reduce_sum,
        'avg': tf.reduce_mean
    }

    # Compute the pooling ratio
    with tf.name_scope('Global{}Pool'.format(pooling_method.capitalize())):
        # Spatial pooling for each image
        def _fn(_in):
            feature_map, shape = _in
            return reducing_op[pooling_method](feature_map[:shape[0], :shape[1]], axis=[0, 1])
        global_pool = tf.map_fn(_fn,
                                (feature_maps, shapes), dtype=tf.float32)
        global_pool.set_shape(feature_maps.get_shape()[0:1].concatenate(feature_maps.get_shape()[3:]))

    if normalize_output:
        # Normalization of the output
        return layers.unit_norm(global_pool, 1)
    else:
        return global_pool


def make_attention_map(queries: tf.Tensor, keys_map: tf.Tensor, keys_map_shapes: tf.Tensor):
    """

    :param queries: [N, Q, F] multiple queries
    :param keys_map: [N, H, W, F] keys map
    :return: [N, H, W, Q]
    """
    assert len(queries.get_shape()) == 3
    assert len(keys_map.get_shape()) == 4

    def _get_shape(tensor: tf.Tensor):
        shape = tf.shape(tensor)
        shape_defined = tensor.get_shape()
        return [shape_defined[i].value if shape_defined[i].value is not None else shape[i] for i in range(len(shape_defined))]

    with tf.name_scope('AttentionMap'):
        N, H, W, F = _get_shape(keys_map)
        reshaped_keys_map = tf.reshape(keys_map, [N, H * W, F])
        result = tf.matmul(reshaped_keys_map, queries, transpose_b=True)
        reshaped_result = tf.reshape(result, [N, H, W, result.get_shape()[-1].value])
        normalized_result = reshaped_result / np.sqrt(F)

    return softmax_2d(normalized_result, keys_map_shapes)


def softmax_2d(logits, shapes):
    """

    :param logits: [N, H, W, Q]
    :param shapes: [N, 2]
    :return: [N, H, W, Q] normalized, and padded with zeroes
    """
    with tf.name_scope('Softmax2d'):
        Q = int(logits.get_shape()[-1])
        padded_shape = tf.shape(logits)  # N, H, W, Q

        def fn(_in):
            attention_map, shape = _in

            attention_map = attention_map[:shape[0], :shape[1], :]
            reshaped_attention_map = tf.reshape(attention_map, [-1, Q])
            reshaped_softmax = tf.nn.softmax(reshaped_attention_map, dim=0)
            softmax = tf.reshape(reshaped_softmax, tf.stack([shape[0], shape[1], Q]))
            padded_softmax = tf.pad(softmax, [[0, padded_shape[1] - shape[0]],
                                              [0, padded_shape[2] - shape[1]],
                                              [0, 0]],
                                    mode='CONSTANT')
            return padded_softmax

        # Spatial pooling for each image
        result = tf.map_fn(fn, (logits, shapes), dtype=tf.float32)
        result.set_shape(logits.get_shape())
    return result


def gather_attention(attention_maps, value_maps, pooling_method: str, normalize_output=True):
    """

    :param attention_maps: [N, H, W, Q]
    :param value_maps: [N, H, W, F]
    :return: [N, Q*F]
    """
    reducing_op = {
        'max': tf.reduce_max,
        'sum': tf.reduce_sum,
        'avg': tf.reduce_mean
    }

    Q, F = int(attention_maps.get_shape()[-1]), int(value_maps.get_shape()[-1])
    with tf.name_scope('AttentionSum'):
        weighted_values = attention_maps[:, :, :, :, np.newaxis] * value_maps[:, :, :, np.newaxis, :]  # N,H,W,Q,F
        result = reducing_op[pooling_method](weighted_values, axis=[1, 2])
        if normalize_output:
            # Normalization of the output
            result = layers.unit_norm(result, 2)
        return tf.reshape(result, [-1, Q*F])