from tensorflow.contrib import slim, layers
import tensorflow as tf
from tensorflow.contrib.slim import nets
from .common import mean_substraction


def resnet_v1_50_fn(input_tensor: tf.Tensor, is_training=False, blocks=4, weight_decay=0.0001, renorm=True) -> tf.Tensor:
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=0.999)), \
         slim.arg_scope([layers.batch_norm], renorm_decay=0.95, renorm=renorm):
        input_tensor = mean_substraction(input_tensor)
        assert 0 < blocks <= 4
        blocks_list = [
              nets.resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              nets.resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              nets.resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
              nets.resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
        ]
        return nets.resnet_v1.resnet_v1(
              input_tensor,
              blocks=blocks_list[:blocks],
              num_classes=None,
              is_training=is_training,
              global_pool=False,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope='resnet_v1_50')[0]