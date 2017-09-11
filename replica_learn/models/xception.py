from tensorflow.contrib import slim, layers
import tensorflow as tf


def xception_fn(input_tensor: tf.Tensor, scope='Xception', blocks=15,
                weight_decay=0.00001, batch_norm=True,
                is_training=False) -> tf.Tensor:
    with tf.variable_scope(scope, 'Xception', [input_tensor]) as sc:
        with tf.name_scope('preprocess'):
            input_tensor = 2 * ((input_tensor / 255.0) - 0.5)
        assert 1 <= blocks <= 15
        end_points_collection = sc.original_name_scope + '_end_points'
        batch_norm_params = {
            'is_training': is_training,
            # 'decay': batch_norm_decay,
            # 'epsilon': batch_norm_epsilon,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([layers.conv2d, layers.fully_connected, layers.max_pool2d],
                            outputs_collections=end_points_collection), \
             slim.arg_scope([layers.conv2d, layers.fully_connected],
                            weights_regularizer=layers.l2_regularizer(weight_decay)), \
             slim.arg_scope([layers.conv2d, layers.separable_conv2d],
                            normalizer_fn=layers.batch_norm if batch_norm else None), \
             slim.arg_scope([layers.batch_norm],
                            **batch_norm_params), \
             slim.arg_scope([layers.separable_conv2d],
                            depth_multiplier=1):

            with tf.variable_scope('block1'):
                net = layers.conv2d(input_tensor, 32, (3, 3), stride=(2, 2), padding='VALID', scope='Conv_1')
                net = layers.conv2d(net, 64, (3, 3), padding='VALID', scope='Conv_2')
                if blocks == 1:
                    return net

            with tf.variable_scope('block2'):
                residual = layers.conv2d(net, 128, (1, 1), stride=(2, 2), activation_fn=None)

                net = layers.separable_conv2d(net, 128, (3, 3), scope='SeparableConv_1')
                net = layers.separable_conv2d(net, 128, (3, 3), activation_fn=None, scope='SeparableConv_2')
                net = layers.max_pool2d(net, (3, 3), stride=(2, 2), padding='same')

                net = net + residual
                if blocks == 2:
                    return net

            with tf.variable_scope('block3'):
                residual = layers.conv2d(net, 256, (1, 1), stride=(2, 2), activation_fn=None)

                net = tf.nn.relu(net)
                net = layers.separable_conv2d(net, 256, (3, 3), scope='SeparableConv_1')
                net = layers.separable_conv2d(net, 256, (3, 3), activation_fn=None, scope='SeparableConv_2')
                net = layers.max_pool2d(net, (3, 3), stride=(2, 2), padding='same')

                net = net + residual
                if blocks == 3:
                    return net

            with tf.variable_scope('block4'):
                residual = layers.conv2d(net, 728, (1, 1), stride=(2, 2), activation_fn=None)

                net = tf.nn.relu(net)
                net = layers.separable_conv2d(net, 728, (3, 3), scope='SeparableConv_1')
                net = layers.separable_conv2d(net, 728, (3, 3), activation_fn=None, scope='SeparableConv_2')
                net = layers.max_pool2d(net, (3, 3), stride=(2, 2), padding='same')

                net = net + residual
                if blocks == 4:
                    return net

            for i in range(8):
                residual = net
                with tf.variable_scope('block{}'.format(i + 5)):
                    net = tf.nn.relu(net)
                    net = layers.separable_conv2d(net, 728, (3, 3), scope='SeparableConv_1')
                    net = layers.separable_conv2d(net, 728, (3, 3), scope='SeparableConv_2')
                    net = layers.separable_conv2d(net, 728, (3, 3), activation_fn=None, scope='SeparableConv_3')

                    net = net + residual
                if blocks == 5 + i:
                    return net

            with tf.variable_scope('block13'):
                residual = layers.conv2d(net, 1024, (1, 1), stride=(2, 2), activation_fn=None)

                net = tf.nn.relu(net)
                net = layers.separable_conv2d(net, 728, (3, 3), scope='SeparableConv_1')
                net = layers.separable_conv2d(net, 1024, (3, 3), activation_fn=None, scope='SeparableConv_2')
                net = layers.max_pool2d(net, (3, 3), stride=(2, 2), padding='same')

                net = net + residual
                if blocks == 13:
                    return net

            with tf.variable_scope('block14'):
                net = layers.separable_conv2d(net, 1536, (3, 3))
                if blocks == 14:
                    return net

            with tf.variable_scope('block15'):
                net = layers.separable_conv2d(net, 2048, (3, 3))
                if blocks == 15:
                    return net
