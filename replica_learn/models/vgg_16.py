from tensorflow.contrib import slim, layers
import tensorflow as tf
from tensorflow.contrib.slim import nets
from .common import mean_substraction


def vgg_16_fn(input_tensor: tf.Tensor, scope='vgg_16', blocks=5, weight_decay=0.0005) -> tf.Tensor:
    with slim.arg_scope(nets.vgg.vgg_arg_scope(weight_decay=weight_decay)):
        with tf.variable_scope(scope, 'vgg_16', [input_tensor]) as sc:
            input_tensor = mean_substraction(input_tensor)
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope(
                    [layers.conv2d, layers.fully_connected, layers.max_pool2d],
                    outputs_collections=end_points_collection):
                net = layers.repeat(
                    input_tensor, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                net = layers.max_pool2d(net, [2, 2], scope='pool1')
                if blocks >= 2:
                    net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                    net = layers.max_pool2d(net, [2, 2], scope='pool2')
                if blocks >= 3:
                    net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
                    net = layers.max_pool2d(net, [2, 2], scope='pool3')
                if blocks >= 4:
                    net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
                    net = layers.max_pool2d(net, [2, 2], scope='pool4')
                if blocks >= 5:
                    net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
                    net = layers.max_pool2d(net, [2, 2], scope='pool5')

                # Convert end_points_collection into a end_point dict.
                # end_points = utils.convert_collection_to_dict(end_points_collection)
                return net