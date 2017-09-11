import tensorflow as tf
import numpy as np


_VGG_MEANS = [123.68, 116.78, 103.94]


def mean_substraction(input_tensor, means=_VGG_MEANS):
    return tf.subtract(input_tensor, np.array(means)[None, None, None, :], name='MeanSubstraction')