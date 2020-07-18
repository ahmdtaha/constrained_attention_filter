# Copyright 2016 pudae. All Rights Reserved.
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
"""Contains the definition of the DenseNet architecture.

As described in https://arxiv.org/abs/1608.06993.

  Densely Connected Convolutional Networks
  Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
slim = tf.contrib.slim
import constants as const
import nets.nn_utils as nn_utils
import utils.os_utils as os_utils
from nets import attention_filter



@slim.add_arg_scope
def _global_avg_pool2d(inputs, data_format='NHWC', scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'xx', [inputs]) as sc:
    axis = [1, 2] if data_format == 'NHWC' else [2, 3]
    net = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


@slim.add_arg_scope
def _conv(inputs, num_filters, kernel_size, stride=1, dropout_rate=None,
          scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'xx', [inputs]) as sc:
    net = slim.batch_norm(inputs)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, num_filters, kernel_size)

    if dropout_rate:
      net = tf.nn.dropout(net)

    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net


@slim.add_arg_scope
def _conv_block(inputs, num_filters, data_format='NHWC', scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'conv_blockx', [inputs]) as sc:
    net = inputs
    net = _conv(net, num_filters*4, 1, scope='x1')
    net = _conv(net, num_filters, 3, scope='x2')
    if data_format == 'NHWC':
      net = tf.concat([inputs, net], axis=3)
    else: # "NCHW"
      net = tf.concat([inputs, net], axis=1)


    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net


@slim.add_arg_scope
def _dense_block(inputs, num_layers, num_filters, growth_rate,
                 grow_num_filters=True, scope=None, outputs_collections=None,filter_type=None,verbose=None):

  with tf.variable_scope(scope, 'dense_blockx', [inputs]) as sc:
    net = inputs
    for i in range(num_layers):
      branch = i + 1
      net = _conv_block(net, growth_rate, scope='conv_block'+str(branch))


      end_point = 'conv_block'+str(branch)
      net = attention_filter.add_attention_filter(net, end_point,verbose=verbose,filter_type=filter_type)

      if grow_num_filters:
        num_filters += growth_rate

    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net, num_filters


@slim.add_arg_scope
def _transition_block(inputs, num_filters, compression=1.0,
                      scope=None, outputs_collections=None):

  num_filters = int(num_filters * compression)
  with tf.variable_scope(scope, 'transition_blockx', [inputs]) as sc:
    net = inputs
    net = _conv(net, num_filters, 1, scope='blk')

    net = slim.avg_pool2d(net, 2)

    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net, num_filters


def densenet(inputs,
             num_classes=1000,
             reduction=None,
             growth_rate=None,
             num_filters=None,
             num_layers=None,
             dropout_rate=None,
             data_format='NHWC',
             is_training=True,
             reuse=None,
             filter_type=None,
             verbose=None,
             scope=None):
  assert reduction is not None
  assert growth_rate is not None
  assert num_filters is not None
  assert num_layers is not None

  compression = 1.0 - reduction
  num_dense_blocks = len(num_layers)

  if data_format == 'NCHW':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  with tf.variable_scope(scope, 'densenetxxx', [inputs, num_classes],
                         reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                         is_training=is_training), \
         slim.arg_scope([slim.conv2d, _conv, _conv_block,
                         _dense_block, _transition_block],
                         outputs_collections=end_points_collection), \
         slim.arg_scope([_conv], dropout_rate=dropout_rate):
      net = inputs

      # initial convolution
      net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv1')
      net = slim.batch_norm(net)
      net = tf.nn.relu(net)
      net = slim.max_pool2d(net, 3, stride=2, padding='SAME')

      # blocks
      for i in range(num_dense_blocks - 1):
        # dense blocks
        net, num_filters = _dense_block(net, num_layers[i], num_filters,
                                        growth_rate,
                                        scope='dense_block' + str(i+1),filter_type=filter_type,verbose=verbose)

        # Add transition_block
        net, num_filters = _transition_block(net, num_filters,
                                             compression=compression,
                                             scope='transition_block' + str(i+1))

      net, num_filters = _dense_block(
              net, num_layers[-1], num_filters,
              growth_rate,
              scope='dense_block' + str(num_dense_blocks),filter_type=filter_type,verbose=verbose)

      # final blocks
      with tf.variable_scope('final_block', [inputs]):
        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        net = _global_avg_pool2d(net, scope='global_avg_pool')

      net = slim.conv2d(net, num_classes, 1,
                        biases_initializer=tf.zeros_initializer(),
                        scope='logits')

      end_points = slim.utils.convert_collection_to_dict(
          end_points_collection)

      if num_classes is not None:
        end_points['predictions'] = slim.softmax(net, scope='predictions')

      return net, end_points


def densenet121(inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
  return densenet(inputs,
                  num_classes=num_classes,
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,24,16],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet121')
densenet121.default_image_size = 224


def densenet161(inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
  return densenet(inputs,
                  num_classes=num_classes,
                  reduction=0.5,
                  growth_rate=48,
                  num_filters=96,
                  num_layers=[6,12,36,24],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet161')
densenet161.default_image_size = 224


def densenet169(inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
  return densenet(inputs,
                  num_classes=num_classes,
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,32,32],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet169')
densenet169.default_image_size = 224


def densenet_arg_scope(weight_decay=1e-4,
                       batch_norm_decay=0.999,
                       batch_norm_epsilon=1e-5,
                       data_format='NHWC'):
  with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.avg_pool2d, slim.max_pool2d,
                       _conv_block, _global_avg_pool2d],
                      data_format=data_format):
    with slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(weight_decay),
                         activation_fn=None,
                         biases_initializer=None):
      with slim.arg_scope([slim.batch_norm],
                          scale=True,
                          decay=batch_norm_decay,
                          epsilon=batch_norm_epsilon) as scope:
        return scope



class DenseNet161:

    def var_2_train(self):
        scopes = [scope.strip() for scope in 'densenet161/logits'.split(',')]
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        # print(variables_to_train)
        return variables_to_train

    def load_model(self,save_model_dir,ckpt_file,sess,saver,load_logits=False):
        # Try to initialize the network from a custom model
        if (os.path.exists(save_model_dir) and os_utils.chkpt_exists(save_model_dir)):
            saver.restore(sess, ckpt_file)
            return 'Model weights initialized from {}'.format(ckpt_file)

        else: # if custom model is not provided, initialize the network weights using imageNet weights
            if(load_logits):
                exclusions = [scope.strip() for scope in '**'.split(',')]
            else:
                exclusions = [scope.strip() for scope in 'global_step,densenet161/logits'.split(',')]

            variables_to_restore = []
            for var in tf.contrib.slim.get_model_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        break
                else:
                    variables_to_restore.append(var)

            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.cfg.imagenet_weights_filepath, variables_to_restore,ignore_missing_vars=False)
            init_fn(sess)

            return 'Model weights initialized from imageNet'

    def __init__(self,cfg, weight_decay=0.0001, data_format='NHWC',reuse=None,
                 images_ph = None,
                 lbls_ph = None,
                 weights_ph=None):

        self.cfg = cfg
        filter_type = cfg.filter_type
        verbose = cfg.print_filter_name
        num_classes = cfg.num_classes
        batch_size = None
        if lbls_ph is not None:
            self.gt_lbls = tf.reshape(lbls_ph,[-1,num_classes])
        else:
            self.gt_lbls = tf.placeholder(tf.int32, shape=(batch_size, num_classes), name='class_lbls')

        self.do_augmentation = tf.placeholder(tf.bool, name='do_augmentation')
        self.loss_class_weight = tf.placeholder(tf.float32, shape=(num_classes, num_classes), name='weights')
        self.input = tf.placeholder(tf.float32, shape=(batch_size, const.max_frame_size, const.max_frame_size,
                                                       const.num_channels), name='context_input')

        # if is_training:
        if images_ph is not None:
            self.input = images_ph
            _,w,h,c = self.input.shape
            aug_imgs = tf.reshape(self.input, [-1, w, h, c])
            print('No nnutils Augmentation')
        else:
            aug_imgs = tf.cond(self.do_augmentation,
                               lambda: nn_utils.augment(self.input,cfg.preprocess_func, horizontal_flip=True, vertical_flip=False,
                                                        rotate=0, crop_probability=0, color_aug_probability=0)
                               , lambda: nn_utils.center_crop(self.input,cfg.preprocess_func))
        # aug_imgs = self.input ## Already augmented



        with tf.contrib.slim.arg_scope(densenet_arg_scope(weight_decay=weight_decay, data_format=data_format)):
            val_nets, val_end_points = densenet(aug_imgs,
                                        num_classes=num_classes,
                                        reduction=0.5,
                                        growth_rate=48,
                                        num_filters=96,
                                        num_layers=[6, 12, 36, 24],
                                        data_format=data_format,
                                        is_training=False,  ## Set is always to false
                                        reuse=None,
                                        filter_type=filter_type,
                                        verbose=verbose,
                                        scope='densenet161')



        def  cal_metrics(end_points):
            gt = tf.argmax(self.gt_lbls, 1)
            logits = tf.reshape(end_points['densenet161/logits'], [-1, num_classes])
            pre_logits = end_points['densenet161/dense_block4']

            center_supervised_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.gt_lbls,
                                                                                         logits=logits,
                                                                                         name='xentropy_center')
            loss = tf.reduce_mean(center_supervised_cross_entropy, name='xentropy_mean')
            predictions = tf.reshape(end_points['predictions'], [-1, num_classes])
            class_prediction = tf.argmax(predictions, 1)
            supervised_correct_prediction = tf.equal(gt, class_prediction)
            supervised_correct_prediction_cast = tf.cast(supervised_correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(supervised_correct_prediction_cast)
            confusion_mat = tf.confusion_matrix(gt, class_prediction, num_classes=num_classes)
            _, accumulated_accuracy = tf.compat.v1.metrics.accuracy(gt, class_prediction)
            _, per_class_acc_acc = tf.compat.v1.metrics.mean_per_class_accuracy(gt, class_prediction, num_classes=num_classes)
            per_class_acc_acc = tf.reduce_mean(per_class_acc_acc)

            class_prediction = tf.nn.softmax(logits)
            return loss,pre_logits,accuracy,confusion_mat,accumulated_accuracy,per_class_acc_acc,class_prediction,logits

        # self.train_loss,self.train_pre_logits,self.train_accuracy,self.train_confusion_mat,\
        # self.train_accumulated_accuracy,self.train_per_class_acc_acc ,self.train_class_prediction,self.train_logits = cal_metrics(train_end_points);

        self.val_loss,self.val_pre_logits,self.val_accuracy, self.val_confusion_mat, self.val_accumulated_accuracy \
            , self.val_per_class_acc_acc ,self.val_class_prediction,self.val_logits = cal_metrics(val_end_points);








