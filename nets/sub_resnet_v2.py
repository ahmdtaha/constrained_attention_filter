# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import constants as const
# import configuration as config
import nets.nn_utils as nn_utils
import utils.os_utils as os_utils

from nets import resnet_utils
from nets import attention_filter

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN before convolutions.

    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
    """Generator for v2 (preactivation) ResNet models.

    This function generates a family of ResNet v2 models. See the resnet_v2_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.

    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.

    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks.
        If 0 or None, we return the features before the logit layer.
      is_training: whether batch_norm layers are in training mode.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
          To use this parameter, the input images must be smaller than 300x300
          pixels, in which case the output logit layer does not contain spatial
          information and can be removed.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.


    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is 0 or None,
        then net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is a non-zero integer, net contains the
        pre-softmax activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                print(net)
                end_point = 'resnet_v2_50'
                # atten_var = tf.get_variable("atten_" + end_point, [net.shape[1], net.shape[2], 1], dtype=tf.float32,
                #                             initializer=tf.contrib.layers.xavier_initializer())
                # print(atten_var)
                # atten_var_norm = atten_var / tf.norm(atten_var)
                # atten_var_gate = tf.Variable(False, name="gate_" + end_point)
                # net = tf.cond(atten_var_gate, lambda: tf.multiply(atten_var_norm, net), lambda: tf.identity(net))
                net = attention_filter.add_attention_filter(net, end_point)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


resnet_v2.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v2 bottleneck block.

    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.

    Returns:
      A resnet_v2 bottleneck block.
    """
    return resnet_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


resnet_v2.default_image_size = 224


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope)


resnet_v2_50.default_image_size = resnet_v2.default_image_size


def resnet_v2_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope)


resnet_v2_101.default_image_size = resnet_v2.default_image_size


def resnet_v2_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope)


resnet_v2_152.default_image_size = resnet_v2.default_image_size


def resnet_v2_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope)


resnet_v2_200.default_image_size = resnet_v2.default_image_size


class ResNet50:
    def var_2_train(self):
        scopes = [scope.strip() for scope in 'resnet_v2_50/logits'.split(',')]
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        print(variables_to_train)
        return variables_to_train;

    def resume_model(self, save_model_dir, ckpt_file, sess, saver):
        variables_to_restore = []
        exclusions = [scope.strip() for scope in '**'.split(',')]
        for var in tf.contrib.slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                variables_to_restore.append(var)

        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(ckpt_file,
                                                                 variables_to_restore, ignore_missing_vars=False)
        init_fn(sess)

    def load_model(self, save_model_dir, ckpt_file, sess, saver, load_logits=False):
        # ckpt_file = tf.train.latest_checkpoint(save_model_dir)
        # if (os.path.exists(save_model_dir) and os_utils.chkpt_exists(save_model_dir)):
        if not ckpt_file is None:
            # Try to restore everything if possible
            saver.restore(sess, ckpt_file)
            return 'Model Loaded Normally';
        else:
            print('Failed to Model Loaded Normally from ', ckpt_file);
            if (load_logits):
                exclusions = [scope.strip() for scope in '**'.split(',')]
            else:
                exclusions = [scope.strip() for scope in 'resnet_v2_50/logits'.split(',')]
            # exclusions = [scope.strip() for scope in '**'.split(',')]
            variables_to_restore = []
            for var in tf.contrib.slim.get_model_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        break
                else:
                    variables_to_restore.append(var)
            # print(variables_to_restore)
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.cfg.imagenet__weights_filepath,
                                                                     variables_to_restore, ignore_missing_vars=False)
            # init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.imagenet__weights_filepath)
            init_fn(sess)
            return 'Failed to Model Loaded Normally from ' + str(
                ckpt_file) + '. Thus, Loaded Some variables loaded from imagenet'

    def __init__(self,
                 cfg=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50',
                 images_ph=None,
                 lbls_ph=None
                 ):
        self.cfg = cfg
        batch_size = None
        if lbls_ph is not None:
            self.gt_lbls = tf.reshape(lbls_ph, [-1, cfg.num_classes])
        else:
            self.gt_lbls = tf.placeholder(tf.int32, shape=(batch_size, cfg.num_classes), name='class_lbls')

        self.do_augmentation = tf.placeholder(tf.bool, name='do_augmentation')
        self.loss_class_weight = tf.placeholder(tf.float32, shape=(cfg.num_classes, cfg.num_classes), name='weights')
        if cfg.db_name == 'honda':
            self.input = tf.placeholder(tf.float32, shape=(batch_size, const.frame_height, const.frame_width,
                                                           const.context_channels), name='context_input')
        else:
            self.input = tf.placeholder(tf.float32, shape=(batch_size, const.max_frame_size, const.max_frame_size,
                                                           const.frame_channels), name='context_input')

        # if is_training:
        if images_ph is not None:
            self.input = images_ph
            _, w, h, c = self.input.shape
            aug_imgs = tf.reshape(self.input, [-1, w, h, c])
            print('No nnutils Augmentation')
        else:
            if cfg.db_name == 'honda':
                aug_imgs = self.input
            else:
                aug_imgs = tf.cond(self.do_augmentation,
                                   lambda: nn_utils.augment(self.input, cfg.preprocess_func, horizontal_flip=True,
                                                            vertical_flip=False,
                                                            rotate=0, crop_probability=0, color_aug_probability=0)
                                   , lambda: nn_utils.center_crop(self.input, cfg.preprocess_func))

        with slim.arg_scope(resnet_arg_scope()):

            _, val_end_points = resnet_v2_50(aug_imgs, cfg.num_classes, is_training=False,
                                             global_pool=global_pool, output_stride=output_stride,
                                             spatial_squeeze=spatial_squeeze,
                                             reuse=True, scope=scope)

        def cal_metrics(end_points):
            gt = tf.argmax(self.gt_lbls, 1);
            logits = tf.reshape(end_points['resnet_v2_50/logits'], [-1, cfg.num_classes])
            pre_logits = None #end_points['resnet_v2_50/block4/unit_3/bottleneck_v2']

            center_supervised_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.gt_lbls,
                                                                                         logits=logits,
                                                                                         name='xentropy_center')
            loss = tf.reduce_mean(center_supervised_cross_entropy, name='xentropy_mean')
            predictions = tf.reshape(end_points['predictions'], [-1, cfg.num_classes])
            class_prediction = tf.argmax(predictions, 1)
            supervised_correct_prediction = tf.equal(gt, class_prediction)
            supervised_correct_prediction_cast = tf.cast(supervised_correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(supervised_correct_prediction_cast)
            confusion_mat = tf.confusion_matrix(gt, class_prediction, num_classes=cfg.num_classes)
            _, accumulated_accuracy = tf.compat.v1.metrics.accuracy(gt, class_prediction)
            _, per_class_acc_acc = tf.compat.v1.metrics.mean_per_class_accuracy(gt, class_prediction, num_classes=cfg.num_classes)
            per_class_acc_acc = tf.reduce_mean(per_class_acc_acc)
            return loss, pre_logits, accuracy, confusion_mat, accumulated_accuracy, per_class_acc_acc, class_prediction,logits

        self.val_loss, self.val_pre_logits, self.val_accuracy, self.val_confusion_mat, \
        self.val_accumulated_accuracy, self.val_per_class_acc_acc, self.val_class_prediction,self.val_logits = cal_metrics(
            val_end_points)





