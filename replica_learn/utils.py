import tensorflow as tf
from typing import Dict
import pickle
import os


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0


def write_as_summaries(training_directory: str, values_dict: Dict[str, float]):
    writer = tf.summary.FileWriter(os.path.join(training_directory, 'eval'))
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=k, simple_value=v) for k, v in values_dict.items()
    ])
    writer.add_summary(summary, _load_global_step_from_checkpoint_dir(training_directory))
    writer.close()


def read_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_pickle(obj, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
