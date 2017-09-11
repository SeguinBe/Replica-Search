import time
import tensorflow as tf
import os
from threading import Thread
from tqdm import tqdm

ENQUEUE_FILENAMES_KEY = 'enqueue_filenames'
DEQUEUE_OUTPUT_KEY = 'dequeue_output'
CLOSE_QUEUE_KEY = 'close_queue'


def signature_def_to_tensors(signature_def, g=None):
    if g is None:
        g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}


def _make_graph(estimator, preprocess_function):
    g = tf.Graph()
    with g.as_default():
        tf.train.create_global_step()
        # random_seed.set_random_seed(estimator._config.tf_random_seed)
        filenames_ph = tf.placeholder(tf.string, shape=[None])
        raw_inputs = tf.map_fn(tf.read_file, filenames_ph, dtype=tf.string)
        resized_images, resized_sizes = tf.map_fn(preprocess_function, raw_inputs,
                                                  dtype=(tf.float32, tf.int32))
        input_features = {'images_1': resized_images, 'image_sizes_1': resized_sizes}

        # Call the model_fn and collect the export_outputs.
        estimator_spec = estimator._call_model_fn(
            features=input_features,
            labels=None,
            mode=tf.estimator.ModeKeys.PREDICT)
        output = estimator_spec.predictions

        # Build the SignatureDefs from receivers and all outputs
        signature_def_params = {
            'predict_from_filenames': filenames_ph,
            'predict_from_encoded_images': raw_inputs,
            # 'predict_from_images': resized_images
        }
        signature_def_map = {
            k: tf.saved_model.signature_def_utils.build_signature_def(
                {'input': tf.saved_model.utils.build_tensor_info(v)},
                {'output': tf.saved_model.utils.build_tensor_info(estimator_spec.predictions['output'])},
                k) for k, v in signature_def_params.items()}

    return g, signature_def_map, estimator_spec


def _make_streaming_graph(estimator, preprocess_function, batch_size=16, num_preprocess_threads=10):
    g = tf.Graph()
    with g.as_default():
        tf.train.create_global_step()
        # random_seed.set_random_seed(estimator._config.tf_random_seed)
        with tf.device('/cpu:0'):
            enqueue_uids = tf.placeholder(tf.string, shape=[None], name='UIDsPlaceholder')
            enqueue_filenames = tf.placeholder(tf.string, shape=[None], name='FilenamesPlaceholder')
            queue = tf.FIFOQueue(capacity=250, dtypes=[tf.string, tf.string], shapes=[[], []],
                                 name='filename_queue')
            filename, uid = queue.dequeue()
            resized_image, image_shape = preprocess_function(tf.read_file(filename))
        input_features = tf.train.batch({'uids': uid,
                                         'images_1': resized_image, 'image_sizes_1': image_shape},
                                        batch_size=batch_size, enqueue_many=False,
                                        allow_smaller_final_batch=True,
                                        capacity=2 * batch_size * num_preprocess_threads,
                                        num_threads=num_preprocess_threads)

        # Call the model_fn and collect the export_outputs.
        estimator_spec = estimator._call_model_fn(
            features=input_features,
            labels=None,
            mode=tf.estimator.ModeKeys.PREDICT)
        output = estimator_spec.predictions

        with tf.control_dependencies([queue.enqueue_many([enqueue_filenames, enqueue_uids])]):
            enqueue_output = tf.shape(enqueue_filenames)[0]
        with tf.control_dependencies([queue.close()]):
            close_queue = queue.size()
        final_outputs = tf.train.batch({'uids': input_features['uids'], **output}, 1, capacity=128,
                                       enqueue_many=True)
        # Dequeue one element and crop the feature_map
        final_outputs = {'uid': final_outputs['uids'][0],
                         'output': final_outputs['output'][0],
                         'feature_map': final_outputs['feature_maps'][0,
                                        :final_outputs['output_shapes'][0, 0],
                                        :final_outputs['output_shapes'][0, 1],
                                        :
                                        ]}
        signature_def_map = {
            ENQUEUE_FILENAMES_KEY: tf.saved_model.signature_def_utils.build_signature_def(
                {'uids': tf.saved_model.utils.build_tensor_info(enqueue_uids),
                 'filenames': tf.saved_model.utils.build_tensor_info(enqueue_filenames)},
                {'output': tf.saved_model.utils.build_tensor_info(enqueue_output)},
                ENQUEUE_FILENAMES_KEY),
            DEQUEUE_OUTPUT_KEY: tf.saved_model.signature_def_utils.build_signature_def(
                None,
                {k: tf.saved_model.utils.build_tensor_info(v) for k, v in final_outputs.items()},
                DEQUEUE_OUTPUT_KEY),
            CLOSE_QUEUE_KEY: tf.saved_model.signature_def_utils.build_signature_def(
                None,
                {'remaining': tf.saved_model.utils.build_tensor_info(close_queue)},
                CLOSE_QUEUE_KEY)
        }

    return g, signature_def_map, estimator_spec


def export_estimator(estimator, export_dir_base, preprocess_function, checkpoint_path=None):

    if not checkpoint_path:
        # Locate the latest checkpoint
        checkpoint_path = tf.train.latest_checkpoint(estimator._model_dir)
    if not checkpoint_path:
        raise ValueError("Couldn't find trained model at %s." % estimator._model_dir)

    def get_timestamped_export_dir(export_dir_base):
        export_timestamp = int(time.time())
        export_dir = os.path.join(export_dir_base, str(export_timestamp))
        return export_dir

    export_dir = get_timestamped_export_dir(export_dir_base)

    # Perform the export
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    g, signature_def_map, estimator_spec = _make_graph(estimator, preprocess_function)
    with tf.Session(graph=g) as session:
        saver_for_restore = estimator_spec.scaffold.saver or tf.train.Saver(
            sharded=True)
        saver_for_restore.restore(session, checkpoint_path)

        builder.add_meta_graph_and_variables(
            session, ['predict'],
            signature_def_map=signature_def_map,
            assets_collection=tf.get_collection(
                tf.GraphKeys.ASSET_FILEPATHS))

    g2, signature_def_map2, _ = _make_streaming_graph(estimator, preprocess_function)
    with tf.Session(graph=g2) as session:
        builder.add_meta_graph(
            ['predict', 'streaming'],
            signature_def_map=signature_def_map2)
    builder.save()

    return export_dir


class LoadedModel:
    def __init__(self, config: tf.ConfigProto, model_dir: str):
        self.config = config
        self.model_dir = model_dir
        self.sess = None

    def __enter__(self):
        with tf.Graph().as_default():
            self.sess = tf.Session(config=self.config)
            loaded_model = tf.saved_model.loader.load(self.sess, ['predict'], self.model_dir)
            self.inputs, self.outputs = signature_def_to_tensors(loaded_model.signature_def['predict_from_filenames'])

    def predict(self, filename: str):
        self.sess.run(self.outputs['output'][0], feed_dict={self.inputs['input']: [filename]})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()


class StreamingModel:
    def __init__(self, config: tf.ConfigProto, model_dir: str):
        self.config = config
        self.model_dir = model_dir
        self.sess = None

    def __enter__(self):
        with tf.Graph().as_default():
            self.sess = tf.Session(config=self.config)
            loaded_model = tf.saved_model.loader.load(self.sess, ['predict', 'streaming'], self.model_dir)
            self.inputs_enq, self.outputs_enq = signature_def_to_tensors(loaded_model.signature_def[ENQUEUE_FILENAMES_KEY])
            _, self.outputs_deq = signature_def_to_tensors(loaded_model.signature_def[DEQUEUE_OUTPUT_KEY])
            _, self.outputs_close = signature_def_to_tensors(loaded_model.signature_def[CLOSE_QUEUE_KEY])
            # Start threads
            self.coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def enqueue(self, uid: str, filename: str):
        self.sess.run(self.outputs_enq, feed_dict={self.inputs_enq['uids']: [uid],
                                                   self.inputs_enq['filenames']: [filename]})

    def threaded_enqueue_many(self, elements, close_queue_when_done=True):
        def enqueueing_fn():
            try:
                for uid, path in elements:
                    self.enqueue(uid, path)
            except Exception as e:
                print(e)
            finally:
                if close_queue_when_done:
                    self.finished_enqueueing()
        Thread(target=enqueueing_fn, daemon=True).start()

    def finished_enqueueing(self):
        print('Closing queue')
        self.sess.run(self.outputs_close)

    def output_generator(self):
        while True:  # while not self.coord.should_stop():
            try:
                yield self.sess.run(self.outputs_deq)
            except tf.errors.OutOfRangeError as e:
                break

    def output_generator_from_iterable(self, elements, disable_tqdm=False):
        """

        :param elements: A iterable of tuples (uid, image_path)
        :param disable_tqdm:
        :return: an iterator of the generated output (a dict with keys 'uid', 'output', 'feature_map', ...)
        """
        self.threaded_enqueue_many(elements)
        return tqdm(self.output_generator(),
                    total=len(elements) if hasattr(elements, '__len__') else None,
                    disable=disable_tqdm)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
