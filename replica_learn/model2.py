import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.ops import state_ops, gen_state_ops
from tensorflow.contrib import rnn
from .utils import ModelParams, TrainingParams

arg_scope = tf.contrib.framework.arg_scope


def authorship_arg_scope(is_training,
                         weight_decay=0.00001,
                         batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,
                         batch_norm_scale=True,
                         ):
    """Defines the default ResNet arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'is_training': is_training,
        'renorm_decay': 0.95,
        'renorm': False,
        'updates_collections': None
    }

    with arg_scope(
            [tf_layers.conv2d, tf_layers.fully_connected],
            weights_regularizer=tf_layers.l2_regularizer(weight_decay),
            activation_fn=tf.nn.relu,
            normalizer_fn=tf_layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([tf_layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def attention_cell(queries: tf.Tensor, attention_keys: tf.Tensor, attention_values: tf.Tensor, class_ids):
    """

    :param queries: [B, C, D]
    :param attention_keys: [N, D]
    :param values: [N, D']
    :param class_ids: [N] with values in 0..C-1
    :return:
    """
    D = queries.get_shape()[-1].value
    assert D == attention_keys.get_shape()[1]
    # Make sparse version of attention_keys
    g = tf.get_default_graph()
    collection_name = 'embedding_class_' + attention_keys.name
    c = tf.get_collection(collection_name)
    # If does not exist
    if len(c) == 0:
        with tf.name_scope('keys_as_sparse'):
            N = tf.shape(attention_keys)[0]
            C = tf.reduce_max(class_ids) + 1
            indices_0 = tf.reshape(tf.tile(tf.range(N)[:, None], tf.stack([1, D])), [-1])
            indices_1 = tf.reshape(tf.tile(class_ids[:, None], tf.stack([1, D])), [-1])
            indices_2 = tf.reshape(tf.tile(tf.range(D)[None, :], tf.stack([N, 1])), [-1])
            indices = tf.cast(tf.stack([indices_0, indices_1 * D + indices_2], axis=1), tf.int64)
            values = tf.reshape(attention_keys, [-1])
            # Note that indices are already in the right-order, so no need for sparse_reorder
            sparse_version = tf.SparseTensor(indices=indices, values=values,
                                             dense_shape=tf.cast(tf.stack([N, C * D]), tf.int64))
            tf.add_to_collection(collection_name, sparse_version)
    else:
        assert len(c) == 1, c
        sparse_version = c[0]

    with tf.name_scope('AttentionCell'):
        # Q*K
        B = tf.shape(queries)[0]
        N = tf.shape(attention_keys)[0]
        C = tf.reduce_max(class_ids) + 1
        # C = tf.Print(C, [B, C, N], 'B,C,N:')
        scores = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_version,
                                                            tf.reshape(queries, tf.stack([B, C * D])),
                                                            adjoint_b=True))  # [B,N]
        scores = scores / np.sqrt(D)
        indices_0 = tf.reshape(tf.range(B)[:, None] * C + class_ids[None, :], [-1])
        indices_1 = tf.reshape(tf.tile(tf.range(N)[None, :], tf.stack([B, 1])), [-1])
        indices = tf.cast(tf.stack([indices_0, indices_1], axis=1), tf.int64)
        sparse_scores = tf.SparseTensor(indices=indices, values=tf.reshape(scores, [-1]),
                                        dense_shape=tf.cast(tf.stack([B * C, N]), tf.int64))
        sparse_scores = tf.sparse_reorder(sparse_scores)
        tf.summary.histogram('attention_scores', sparse_scores.values)
        sparse_scores = tf.sparse_softmax(sparse_scores)
        tf.summary.histogram('attention_scores_sofmax', sparse_scores.values)

        # scores*V
        return tf.reshape(tf.sparse_tensor_dense_matmul(sparse_scores, attention_values),
                          [B, C, attention_values.get_shape()[-1]])


def attention_cell_2(queries: tf.Tensor, attention_keys: tf.Tensor, attention_values: tf.Tensor, class_ids):
    """

    :param queries: [B, C, D]
    :param attention_keys: [N, D]
    :param values: [N, D']
    :param class_ids: [N] with values in 0..C-1
    :return:
    """
    with tf.name_scope('AttentionCell2'):
        # Q*K
        get_scores = True
        N = attention_keys.get_shape().as_list()[0]
        if N is None:
            N = tf.shape(attention_keys)[0]
            get_scores = False
        B = queries.get_shape().as_list()[0]
        if B is None:
            B = tf.shape(queries)[0]
            get_scores = False
        C = tf.reduce_max(class_ids) + 1

        if get_scores:
            # Get only scores for the first image in batch, for debug purposes
            all_scores_transposed_var = gen_state_ops._temporary_variable(shape=[N, B], dtype=tf.float32)
            var_name = all_scores_transposed_var.op.name
            all_scores_transposed_var_initialized = state_ops.assign(all_scores_transposed_var, tf.zeros([N, B]))

        #queries = tf.Print(queries, [tf.shape(queries)], 'entering')

        def single_class_attention(_input):
            single_class_queries, class_id = _input
            mask = tf.equal(class_ids, class_id)
            selected_keys = tf.boolean_mask(attention_keys, mask)  # [N',D]
            selected_values = tf.boolean_mask(attention_values, mask)  # [N',D]
            scores = tf.matmul(single_class_queries, selected_keys, transpose_b=True)  # [B, N']
            scores = tf.nn.softmax(scores)  # [B, N']
            #scores = tf.Print(scores, [tf.shape(scores), tf.shape(selected_values)], 'computing')
            if get_scores:
                update_score_op = [tf.scatter_add(all_scores_transposed_var, tf.where(mask)[:, 0],
                                                  tf.transpose(scores))]
                with tf.control_dependencies(update_score_op):
                    return tf.matmul(scores, selected_values)  # [B, D']
            else:
                return tf.matmul(scores, selected_values)  # [B, D']

        with tf.control_dependencies([all_scores_transposed_var_initialized] if get_scores else []):
            result_transposed = tf.map_fn(single_class_attention,
                                          [tf.transpose(queries, [1, 0, 2]), tf.range(C)], tf.float32,
                                          parallel_iterations=10)  # [C, B, D']
        result_transposed.set_shape((None, None, attention_values.get_shape().as_list()[-1]))
        if get_scores:
            with tf.control_dependencies([result_transposed]):
                all_scores = tf.transpose(gen_state_ops._destroy_temporary_variable(all_scores_transposed_var,
                                                                       var_name=all_scores_transposed_var.op.name),
                                          name='AttentionScoresSoftmax')
                tf.add_to_collection('attention_maps', all_scores)
            with tf.control_dependencies([all_scores]):
                result = tf.transpose(result_transposed, [1, 0, 2])
            # tf.summary.histogram('attention_scores_sofmax', all_scores)
        else:
            result = tf.transpose(result_transposed, [1, 0, 2])
        return result


def full_model_authorship_set_fn(model_params: ModelParams, training_params: TrainingParams):
    def fn(input_tensor, class_ids, image_embeddings, is_training):
        """

        :param input_tensor: [B, F]
        :param class_ids: [N] of elements in [0..C)
        :param image_embeddings: [N, F]
        :param is_training:
        :return:  [B, C] scores
        """
        with tf.name_scope('shapes'):
            C = tf.reduce_max(class_ids) + 1
            B = input_tensor.get_shape().as_list()[0] or tf.shape(input_tensor)[0]
            N = image_embeddings.get_shape().as_list()[0] or tf.shape(image_embeddings)[0]
        # image_embeddings = tf.constant(params['image_embeddings'], dtype=tf.float32, name='image_embeddings')
        first_queries = tf.tile(input_tensor[:, None, :], [1, C, 1])  # [B, C, F]

        lstm_cell = rnn.BasicLSTMCell(model_params.lstm_size)

        hidden_state = tf.ones(tf.stack([B*C, model_params.lstm_size]))
        current_state = tf.zeros(tf.stack([B*C, model_params.lstm_size]))
        state = hidden_state, current_state
        # image_vectors = tf.nn.embedding_lookup(image_embeddings, class_ids)  # [N, F]
        with arg_scope(authorship_arg_scope(is_training=is_training)):

            for i in range(model_params.lstm_steps):
                with tf.variable_scope('Queries_{}'.format(i)):
                    queries = tf_layers.fully_connected(first_queries, model_params.lstm_size//2, activation_fn=None)
                with tf.variable_scope('Keys_{}'.format(i)):
                    keys = tf_layers.fully_connected(image_embeddings, model_params.lstm_size, activation_fn=None)
                with tf.variable_scope('Values_{}'.format(i)):
                    values = tf_layers.fully_connected(image_embeddings, model_params.lstm_size//2, activation_fn=None)

                att_queries = tf.reshape(state[0], [B, C, model_params.lstm_size])

                att_result = attention_cell_2(att_queries, keys, values, class_ids)  # [B, C, F]
                next_input = tf.concat([queries, att_result], axis=-1)
                output, state = lstm_cell(tf.reshape(next_input, [-1, model_params.lstm_size]), state)

            hidden = tf_layers.fully_connected(tf.reshape(output, [B, C, model_params.lstm_size]), 128)

            logits = tf_layers.fully_connected(hidden, 1, activation_fn=None, normalizer_fn=None)[:, :, 0]  # [B, C]
            tf.summary.histogram('logits', logits)
        return logits

    return fn


def full_model_authorship_set_fn3(params):
    def fn(input_tensor, class_ids, image_embeddings, is_training):
        """

        :param input_tensor: [B, F]
        :param class_ids: [N] of elements in [0..C)
        :param image_embeddings: [N, F]
        :param is_training:
        :return:  [B, C] scores
        """
        C = tf.reduce_max(class_ids) + 1
        # image_embeddings = tf.constant(params['image_embeddings'], dtype=tf.float32, name='image_embeddings')
        first_queries = tf.tile(input_tensor[:, None, :], [1, C, 1])  # [B, C, F]
        # image_vectors = tf.nn.embedding_lookup(image_embeddings, class_ids)  # [N, F]
        with arg_scope(authorship_arg_scope(is_training=is_training)):

            with tf.variable_scope('test'):
                first_values = tf_layers.fully_connected(image_embeddings, 512, activation_fn=None)

            with tf.variable_scope('test', reuse=True):
                first_queries = tf_layers.fully_connected(first_queries, 512, activation_fn=None)

            #first_values = image_embeddings

            att_result = attention_cell_2(first_queries, first_values, first_values, class_ids)  # [B, C, F]

            next_queries = tf_layers.fully_connected(att_result, 512, activation_fn=None)
            next_values = tf_layers.fully_connected(image_embeddings, 512, activation_fn=None)

            att_result_2 = attention_cell_2(next_queries, next_values, next_values, class_ids)  # [B, C, F]

            logits = tf_layers.fully_connected(att_result_2, 1, activation_fn=None, normalizer_fn=None)[:, :, 0]  # [B, C]
            tf.summary.histogram('logits', logits)
        return logits

    return fn


def full_model_authorship_set_fn2(params):
    def fn(input_tensor, class_ids, class_nb_elements, image_embeddings, is_training):
        # image_embeddings = tf.constant(params['image_embeddings'], dtype=tf.float32, name='image_embeddings')

        image_vectors = tf.nn.embedding_lookup(image_embeddings, class_ids)  # [N, M, F]

        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in params['lstm_sizes']]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, sequence_length=class_nb_elements,
                                           inputs=image_vectors,
                                           dtype=tf.float32)

        # if is_training:
        #     outputs = tf.nn.dropout(outputs, keep_prob=0.8)
        class_embeddings = tf.gather_nd(outputs,
                                        tf.stack([tf.range(tf.shape(class_ids)[0]), class_nb_elements - 1],
                                                 axis=1))  # [C,F]

        with tf.variable_scope('MergeLayer'):
            logits = tf.matmul(input_tensor, class_embeddings, transpose_b=True)
            logits = tf.nn.bias_add(logits,
                                    tf.get_variable('bias', shape=[class_ids.get_shape()[0]],
                                                    initializer=tf.zeros_initializer()))

        return logits

    return fn


def model_authorship_fn(features, labels, mode, params):
    """

    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """

    image_embeddings_values = params['image_embeddings']
    class_ids_values = params['class_ids']
    model_params = params['model_params']  # type: ModelParams
    training_params = params['training_params']  # type: TrainingParams

    # image_embeddings = tf.constant(params['image_embeddings'], dtype=tf.float32, name='image_embeddings')
    #print(params['image_embeddings'].shape)
    image_embeddings = tf.get_variable(name='image_embeddings', shape=image_embeddings_values.shape,
                                       trainable=False, dtype=tf.float32)
    image_embeddings_ph = tf.placeholder(tf.float32)
    image_embeddings_assign_op = tf.assign(image_embeddings, image_embeddings_ph)
    default_class_ids = tf.get_variable(name='class_ids', shape=class_ids_values.shape,
                                       trainable=False, dtype=tf.int32)
    default_class_ids_ph = tf.placeholder(tf.int32)
    default_class_ids_assign_op = tf.assign(default_class_ids, default_class_ids_ph)

    if 'input_ids' in features:
        input_ids = features['input_ids']
        input_tensor = tf.nn.embedding_lookup(image_embeddings, input_ids)
    else:
        input_tensor = features['inputs']
    if 'class_ids' in features:
        class_ids = features['class_ids']
    else:
        class_ids = default_class_ids
    # class_nb_elements = features['class_nb_elements']

    full_model = full_model_authorship_set_fn(model_params, training_params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        to_be_used_class_ids = tf.not_equal(class_ids, -1)
        image_embeddings = tf.boolean_mask(image_embeddings, to_be_used_class_ids)
        class_ids = tf.boolean_mask(class_ids, to_be_used_class_ids)

    logits = full_model(input_tensor, class_ids,
                        image_embeddings=image_embeddings,
                        is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        one_hot = tf.one_hot(labels, tf.reduce_max(class_ids) + 1)
        if 'class_weights' in params:
            # TODO reactivate
            # weights = tf.matmul(one_hot, tf.constant(params['class_weights'][:, None].astype(np.float32)))
            # loss = tf.losses.softmax_cross_entropy(one_hot, logits, weights[:, 0])
            loss = tf.losses.softmax_cross_entropy(one_hot, logits)
        else:
            loss = tf.losses.softmax_cross_entropy(one_hot, logits)
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:

        def init_fn(scaffold, session):
            session.run(image_embeddings_assign_op, feed_dict={image_embeddings_ph: image_embeddings_values})
            session.run(default_class_ids_assign_op, feed_dict={default_class_ids_ph: class_ids_values})

        # TODO Understand why that poses problems
        #ema = tf.train.ExponentialMovingAverage(0.95)
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([loss]))
        #ema_loss = ema.average(loss)
        #tf.summary.scalar('losses/loss', ema_loss)

        with tf.name_scope('Losses'):
            regularization_loss = tf.losses.get_regularization_loss()
            total_loss = loss + regularization_loss
        # total_loss = tf.Print(total_loss,[total_loss], 'total_loss')

        tf.summary.scalar('losses/loss_per_batch', loss)

        tf.summary.scalar('losses/regularization_loss', regularization_loss)

        # Train op
        learning_rate = tf.train.exponential_decay(training_params.learning_rate,
                                                   tf.train.get_or_create_global_step(),
                                                   decay_rate=training_params.lr_decay_rate,
                                                   decay_steps=training_params.lr_decay_steps,
                                                   )
        tf.summary.scalar('train/learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(total_loss,
                                          global_step=tf.train.get_or_create_global_step()
                                          )
            # train_op = tf.Print(train_op, [], message="training op")
    else:
        train_op, init_fn = None, None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.nn.softmax(logits)
        attention_maps = tf.get_collection('attention_maps')
        predictions = {
            'predictions': predictions,
            **{
                'attention_map_{}'.format(i): m for i, m in enumerate(attention_maps)
            }
        }
        if 'uids' in features.keys():
            predictions['uids'] = features['uids']
    else:
        predictions = None

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'top_1': tf.metrics.recall_at_k(labels, logits, 1),
                   'top_3': tf.metrics.recall_at_k(labels, logits, 3),
                   'top_5': tf.metrics.recall_at_k(labels, logits, 5),
                   #'MCA': tf.metrics.mean_per_class_accuracy(labels, tf.argmax(logits, axis=1),
                   #                                          params['n_classes'])
                   }
    else:
        metrics = None

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions,
                                      eval_metric_ops=metrics,
                                      loss=loss,
                                      train_op=train_op,
                                      scaffold=tf.train.Scaffold(init_fn=init_fn)
                                      )
