import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from .layers import global_pool_layer, make_attention_map, gather_attention
from .models import resnet_v1_50_fn, vgg_16_fn, xception_fn


def full_model_fn(params):
    def fn(input_tensor, input_shapes, is_training, output_feature_maps=False):
        base_model_fn = {
            'vgg16': lambda _in: vgg_16_fn(_in, weight_decay=0.0001, blocks=params['blocks']),
            'resnet50': lambda _in: resnet_v1_50_fn(_in, is_training and params['train_batch_norm'],
                                                    blocks=params['blocks'], weight_decay=params['weight_decay']),
            'xception': lambda _in: xception_fn(_in, is_training=is_training and params['train_batch_norm'],
                                                blocks=params['blocks']),
            'xception_fused': lambda _in: xception_fn(_in, batch_norm=False,
                                                      blocks=params['blocks'])
        }
        feature_maps = base_model_fn[params['base_model']](input_tensor)

        with tf.name_scope('OutputShape'):
            increment = tf.shape(input_tensor)[1] / tf.shape(feature_maps)[1]
            result_shapes = tf.cast(tf.round(tf.cast(input_shapes, tf.float64) / increment), tf.int32)

        global_pool = global_pool_layer(feature_maps, result_shapes, params['reducing_op'], normalize_output=True)

        if output_feature_maps:
            return {'feature_maps': feature_maps, 'output_shapes': result_shapes, 'output': global_pool}
        else:
            return {'output': global_pool}

    return fn


def matcher_fn(params):
    def fn(feature_maps1, feature_maps_shape1, feature_maps2, feature_maps_shape2, is_training):
        with tf.name_scope(None, 'Matcher'):
            attention_results1_2, attention_results2_1 = feature_maps2, feature_maps1
            for _ in range(params['matcher_params']['nb_attention_layers']):
                with tf.name_scope(None, 'Attention1_2'):
                    queries1 = global_pool_layer(attention_results2_1, feature_maps_shape1, params['reducing_op'],
                                                 normalize_output=True)[:, np.newaxis, :]
                    attention_map2 = make_attention_map(queries1, attention_results1_2, feature_maps_shape2)
                    attention_results1_2_new = gather_attention(attention_map2, attention_results1_2, 'sum')
                with tf.name_scope(None, 'Attention2_1'):
                    queries2 = global_pool_layer(attention_results1_2, feature_maps_shape2, params['reducing_op'],
                                                 normalize_output=True)[:, np.newaxis, :]
                    attention_map1 = make_attention_map(queries2, attention_results2_1, feature_maps_shape1)
                    attention_results2_1_new = gather_attention(attention_map1, attention_results2_1, 'sum')
                attention_results1_2, attention_results2_1 = attention_results1_2_new, attention_results2_1_new
                if is_training:
                    tf.summary.image('AttentionMap1', attention_map1[:, :, :, 0:1], max_outputs=1)
                    tf.summary.image('AttentionMap2', attention_map2[:, :, :, 0:1], max_outputs=1)
            with tf.name_scope(None, 'Correlation'):
                return tf.reduce_sum(attention_results1_2 * attention_results2_1, axis=1, keep_dims=True)

    return fn


def model_fn(features, labels, mode, params):
    """

    :param features:
    :param labels:
    :param mode:
    :param params: 
    :return:
    """
    full_model = full_model_fn(params)
    matcher = matcher_fn(params)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

        if 'images_3' in features.keys():  # TRIPLET VERSION
            all_images = tf.concat([features['images_1'], features['images_2'], features['images_3']], axis=0)
            # all_images = tf.Print(all_images, [tf.shape(all_images)])
            all_sizes = tf.concat([features['image_sizes_1'], features['image_sizes_2'], features['image_sizes_3']],
                                  axis=0)
            network_outputs = full_model(all_images, all_sizes, is_training=True, output_feature_maps=True)
            # all_features = tf.Print(all_features, [all_features])
            all_features = network_outputs['output']
            feature_maps = network_outputs['feature_maps']
            feature_maps_shape = network_outputs['output_shapes']

            with tf.name_scope('Features1'):
                nb_images_1 = tf.shape(features['images_1'])[0]
                feats1 = all_features[:nb_images_1, :]
                feature_maps1 = feature_maps[:nb_images_1, :, :, :]
                feature_maps_shape1 = feature_maps_shape[:nb_images_1, :]

            with tf.name_scope('Features2'):
                nb_images_2 = tf.shape(features['images_2'])[0]
                feats2 = all_features[nb_images_1:nb_images_1 + nb_images_2, :]
                feature_maps2 = feature_maps[nb_images_1:nb_images_1 + nb_images_2, :, :, :]
                feature_maps_shape2 = feature_maps_shape[nb_images_1:nb_images_1 + nb_images_2, :]

            with tf.name_scope('Features3'):
                nb_images_3 = tf.shape(features['images_3'])[0]
                feats3 = all_features[nb_images_1 + nb_images_2:nb_images_1 + nb_images_2 + nb_images_3, :]
                feature_maps3 = feature_maps[nb_images_1 + nb_images_2:nb_images_1 + nb_images_2 + nb_images_3, :, :, :]
                feature_maps_shape3 = feature_maps_shape[
                                      nb_images_1 + nb_images_2:nb_images_1 + nb_images_2 + nb_images_3, :]

            # TODO standardize the simple version with the matcher version
            # corr_tensor = matcher(feature_maps1, feature_maps_shape1, feature_maps2, feature_maps_shape2,
            #                      is_training=True)
            # corr_tensor_neg = matcher(feature_maps1, feature_maps_shape1, feature_maps3, feature_maps_shape3,
            #                          is_training=True)

            def correlation_function(feats1, feats2) -> tf.Tensor:
                with tf.name_scope(None, 'Correlation'):
                    return tf.reduce_sum(feats1 * feats2, axis=1, keep_dims=True)

            corr_tensor = correlation_function(feats1, feats2)
            corr_tensor_neg = correlation_function(feats1, feats3)

            # corr_tensor = tf.Print(corr_tensor, [corr_tensor])
            # corr_tensor_neg = tf.Print(corr_tensor_neg, [corr_tensor_neg])

            # Check that the shape are just the batch or possibly extended with 1's
            assert len(corr_tensor.get_shape()) == 1 or all([d == 1 for d in corr_tensor.get_shape()[1:]])
            assert len(corr_tensor_neg.get_shape()) == 1 or all([d == 1 for d in corr_tensor.get_shape()[1:]])
            with tf.name_scope('TripletLoss'):
                loss = tf.reduce_mean(tf.maximum(0.0, corr_tensor_neg - corr_tensor + params['triplet_loss_margin']),
                                      name='TripletLoss')
        elif 'images_2' in features.keys():  # SIAMESE VERSION
            feats1 = full_model(features['images_1'], features['image_sizes_1'], is_training=True)['output']
            with tf.variable_scope('', reuse=True):
                feats2 = full_model(features['images_2'], features['image_sizes_2'], is_training=True)['output']
            with tf.name_scope('Correlation'):
                corr_tensor = tf.reduce_sum(feats1 * feats2, axis=1, keep_dims=True)
            with tf.name_scope('ContrastiveLoss'):
                d_squared = 2.0 * (1.0 - corr_tensor)
                # d_squared = tf.reduce_sum(tf.square(feats1-feats2), axis=-1, keep_dims=True)
                loss = tf.reduce_mean(labels * d_squared +
                                      (1 - labels) * tf.square(
                                          tf.maximum(0.0, params['contrastive_loss_margin'] - tf.sqrt(d_squared))),
                                      name='ContrastiveLoss')
                tf.summary.histogram('D_squared/positives', tf.boolean_mask(d_squared, tf.greater(labels, 0.)))
                tf.summary.histogram('D_squared/negatives', tf.boolean_mask(d_squared, tf.equal(labels, 0.)))
        else:
            raise NotImplementedError
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Pretrained weights as initialization
        pretrained_restorer = tf.train.Saver(var_list=[v for v in tf.global_variables()
                                                       if 'global_step' not in v.name and '/renorm' not in v.name
                                                       and 'generalized_mean_p' not in v.name])

        def init_fn(scaffold, session):
            pretrained_restorer.restore(session, params['pretrained_file'])

        # TODO Understand why that poses problems
        #ema = tf.train.ExponentialMovingAverage(0.95)
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([loss]))
        #ema_loss = ema.average(loss)
        #tf.summary.scalar('losses/loss_ema', ema_loss)

        regularization_loss = tf.losses.get_regularization_loss()
        total_loss = loss + regularization_loss
        # total_loss = tf.Print(total_loss,[total_loss], 'total_loss')

        tf.summary.scalar('losses/loss_per_batch', loss)

        tf.summary.scalar('losses/regularization_loss', regularization_loss)

        with tf.name_scope('ImageSummaries'):
            tf.summary.image('input/images_1', tf.image.resize_images(features['images_1'],
                                                                      tf.cast(tf.shape(features['images_1'])[1:3] / 3,
                                                                              tf.int32)), max_outputs=1)
            if 'images_2' in features.keys():
                tf.summary.image('input/images_2', tf.image.resize_images(features['images_2'],
                                                                          tf.cast(
                                                                              tf.shape(features['images_2'])[1:3] / 3,
                                                                              tf.int32)), max_outputs=1)
            if 'images_3' in features.keys():
                tf.summary.image('input/images_3', tf.image.resize_images(features['images_3'],
                                                                          tf.cast(
                                                                              tf.shape(features['images_3'])[1:3] / 3,
                                                                              tf.int32)), max_outputs=1)

        # Train op
        learning_rate = tf.train.exponential_decay(params['learning_rate'],
                                                   tf.train.get_or_create_global_step(),
                                                   decay_rate=params['decay_rate'],
                                                   decay_steps=params['decay_steps'],
                                                   )
        tf.summary.scalar('train/learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(total_loss,
                                          global_step=tf.train.get_or_create_global_step())
            # train_op = tf.Print(train_op, [], message="training op")
    else:
        train_op, init_fn = None, None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = full_model(features['images_1'], features['image_sizes_1'], is_training=False,
                                 output_feature_maps=True)
        if 'uids' in features.keys():
            predictions['uids'] = features['uids']
    else:
        predictions = None

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      scaffold=tf.train.Scaffold(init_fn=init_fn)
                                      )


def full_model_authorship_fn(params):
    def fn(input_tensor, input_shapes, is_training):
        base_model_fn = {
            'vgg16': lambda _in: vgg_16_fn(_in, weight_decay=0.0001, blocks=params['blocks']),
            'resnet50': lambda _in: resnet_v1_50_fn(_in, is_training and params['train_batch_norm'],
                                                    blocks=params['blocks']),
            'xception': lambda _in: xception_fn(_in, is_training=is_training and params['train_batch_norm'],
                                                blocks=params['blocks']),
            'xception_fused': lambda _in: xception_fn(_in, batch_norm=False,
                                                      blocks=params['blocks'])
        }
        feature_maps = base_model_fn[params['base_model']](input_tensor)

        with tf.name_scope('OutputShape'):
            increment = tf.shape(input_tensor)[1] / tf.shape(feature_maps)[1]
            result_shapes = tf.cast(tf.round(tf.cast(input_shapes, tf.float64) / increment), tf.int32)

        global_pool = global_pool_layer(feature_maps, result_shapes, params['reducing_op'], normalize_output=False)

        class_embedding_dim = params['class_embedding_dim']
        if class_embedding_dim > 0:
            embeddings_voc = tf.get_variable('class_embeddings', [params['n_classes'], class_embedding_dim])

            def merge(features, embeddings, activation_fn=tf.nn.relu):
                with tf.variable_scope('MergeLayer'):
                    in_units = features.get_shape()[-1]
                    out_units = embeddings.get_shape()[0]
                    weights = tf.layers.dense(embeddings_voc, in_units, activation=None)  # [C, U]
                    #pb = tf.layers.dense(embeddings, units, activation=None)  # [C, U]
                    #c = pa[:, None, :] + pb[None, :, :]
                    c = tf.matmul(global_pool, weights, transpose_b=True)
                    c = tf.nn.bias_add(c, tf.get_variable('bias', shape=[out_units], initializer=tf.zeros_initializer()))
                    if activation_fn is not None:
                        c = activation_fn(c)
                    return c

            return merge(global_pool, embeddings_voc, activation_fn=None)

            # return tf_layers.fully_connected(internal_fc, num_outputs=1, activation_fn=None)[:, :, 0]

        else:
            return tf_layers.fully_connected(global_pool, num_outputs=params['n_classes'], activation_fn=None)

    return fn


def model_authorship_fn(features, labels, mode, params):
    """

    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """
    full_model = full_model_authorship_fn(params)

    logits = full_model(features['images'], features['image_sizes'],
                        is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        one_hot = tf.one_hot(labels, params['n_classes'])
        if 'class_weights' in params:
            weights = tf.matmul(one_hot, tf.constant(params['class_weights'][:, None].astype(np.float32)))
            loss = tf.losses.softmax_cross_entropy(one_hot, logits, weights[:, 0])
        else:
            loss = tf.losses.softmax_cross_entropy(one_hot, logits)
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Pretrained weights as initialization
        pretrained_restorer = tf.train.Saver(var_list=[v for v in tf.global_variables()
                                                       if params[
                                                           'pretrained_name_scope'] in v.name and '/renorm' not in v.name])

        def init_fn(scaffold, session):
            pretrained_restorer.restore(session, params['pretrained_file'])

        # TODO Understand why that poses problems
        # ema = tf.train.ExponentialMovingAverage(0.95)
        # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([loss]))
        # ema_loss = ema.average(loss)
        # tf.summary.scalar('losses/loss', ema_loss)

        regularization_loss = tf.losses.get_regularization_loss()
        total_loss = loss + regularization_loss
        # total_loss = tf.Print(total_loss,[total_loss], 'total_loss')

        tf.summary.scalar('losses/loss_per_batch', loss)

        tf.summary.scalar('losses/regularization_loss', regularization_loss)

        with tf.name_scope('ImageSummaries'):
            tf.summary.image('input/images', tf.image.resize_images(features['images'],
                                                                    tf.cast(tf.shape(features['images'])[1:3] / 3,
                                                                            tf.int32)), max_outputs=1)

        # Train op
        learning_rate = tf.train.exponential_decay(params['learning_rate'],
                                                   tf.train.get_or_create_global_step(),
                                                   decay_rate=params['decay_rate'],
                                                   decay_steps=params['decay_steps'],
                                                   )
        tf.summary.scalar('train/learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(total_loss,
                                          global_step=tf.train.get_or_create_global_step())
            # train_op = tf.Print(train_op, [], message="training op")
    else:
        train_op, init_fn = None, None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = logits
        if 'uids' in features.keys():
            predictions['uids'] = features['uids']
    else:
        predictions = None

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'top_1': tf.metrics.recall_at_k(labels, logits, 1),
                   'top_3': tf.metrics.recall_at_k(labels, logits, 3),
                   'top_5': tf.metrics.recall_at_k(labels, logits, 5),
                   'MCA': tf.metrics.mean_per_class_accuracy(labels, tf.argmax(logits, axis=1),
                                                             params['n_classes'])}
    else:
        metrics = None

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=logits,
                                      eval_metric_ops=metrics,
                                      loss=loss,
                                      train_op=train_op,
                                      scaffold=tf.train.Scaffold(init_fn=init_fn)
                                      )
