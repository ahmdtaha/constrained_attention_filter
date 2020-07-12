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
"""Contains the definition for inception v1 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import constants as const
from nets import inception_utils
import nets.nn_utils as nn_utils
import utils.os_utils as os_utils
from nets import attention_filter

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v1_base(inputs,
                      final_endpoint='Mixed_5c',
                      include_root_block=True,
                      scope='InceptionV1'):
  """Defines the Inception V1 base architecture.

  This architecture is defined in:
    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']. If
      include_root_block is False, ['Conv2d_1a_7x7', 'MaxPool_2a_3x3',
      'Conv2d_2b_1x1', 'Conv2d_2c_3x3', 'MaxPool_3a_3x3'] will not be available.
    include_root_block: If True, include the convolution and max-pooling layers
      before the inception modules. If False, excludes those layers.
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  end_points = {}
  with tf.variable_scope(scope, 'InceptionV1', [inputs]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=trunc_normal(0.01)):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        net = inputs
        if include_root_block:
          end_point = 'Conv2d_1a_7x7'
          net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point)
          end_points[end_point] = net
          if final_endpoint == end_point:
            return net, end_points
          end_point = 'MaxPool_2a_3x3'
          net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
          end_points[end_point] = net
          if final_endpoint == end_point:
            return net, end_points
          end_point = 'Conv2d_2b_1x1'
          net = slim.conv2d(net, 64, [1, 1], scope=end_point)
          end_points[end_point] = net
          if final_endpoint == end_point:
            return net, end_points
          end_point = 'Conv2d_2c_3x3'
          net = slim.conv2d(net, 192, [3, 3], scope=end_point)
          end_points[end_point] = net
          if final_endpoint == end_point:
            return net, end_points
          end_point = 'MaxPool_3a_3x3'
          net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
          end_points[end_point] = net
          if final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])


        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_5a_2x2'
        net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        # print(net)
        # atten_var = tf.get_variable("atten_" + end_point, [net.shape[1], net.shape[2], 1], dtype=tf.float32,
        #                             initializer=tf.contrib.layers.xavier_initializer())
        # print(atten_var)
        # atten_var_norm = atten_var / tf.norm(atten_var)
        # atten_var_gate = tf.Variable(False, name="gate_" + end_point)
        # net = tf.cond(atten_var_gate, lambda: tf.multiply(atten_var_norm, net), lambda: tf.identity(net))
        net = attention_filter.add_attention_filter(net,end_point,verbose=True)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1',
                 global_pool=False):
  """Defines the Inception V1 architecture.

  This architecture is defined in:

    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
        shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  # Final pooling and prediction
  with tf.variable_scope(scope, 'InceptionV1', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v1_base(inputs, scope=scope)



      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
          end_points['AvgPool_0a_7x7'] = net
        if not num_classes:
          return net, end_points
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout_0b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_0c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
inception_v1.default_image_size = 224

inception_v1_arg_scope = inception_utils.inception_arg_scope



class InceptionV1:

    def var_2_train(self):
        scopes = [scope.strip() for scope in 'InceptionV1/Logits'.split(',')]
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        print(variables_to_train)
        return variables_to_train;

    def resume_model(self,save_model_dir,ckpt_file,sess,saver):
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

    def load_model(self,save_model_dir,ckpt_file,sess,saver,load_logits=False):
        if (os.path.exists(save_model_dir) and os_utils.chkpt_exists(save_model_dir)):
            # Try to restore everything if possible
            saver.restore(sess, ckpt_file)
            return 'Model weights initialized from {}'.format(ckpt_file)
        else:
            if (load_logits):
                exclusions = [scope.strip() for scope in '**'.split(',')]
            else:
                exclusions = [scope.strip() for scope in 'Logits,InceptionV1/Logits,InceptionV1/AuxLogits'.split(',')]
            # exclusions = [scope.strip() for scope in '**'.split(',')]
            variables_to_restore = []
            for var in tf.contrib.slim.get_model_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        break
                else:
                    variables_to_restore.append(var)
            # print(variables_to_restore)
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.cfg.imagenet_weights_filepath, variables_to_restore,ignore_missing_vars=False)
            # init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.imagenet__weights_filepath)
            init_fn(sess)
            return 'Model weights initialized from imageNet'


    def __init__(self, cfg=None, is_training=True,
                 dropout_keep_prob=0.8,
                 scope='InceptionV1',
                 images_ph=None,
                 lbls_ph=None
                 ):
        self.cfg = cfg
        batch_size = None
        num_classes = cfg.num_classes
        if lbls_ph is not None:
            self.gt_lbls = tf.reshape(lbls_ph, [-1, num_classes])
        else:
            self.gt_lbls = tf.placeholder(tf.int32, shape=(batch_size, num_classes), name='class_lbls')

        self.do_augmentation = tf.placeholder(tf.bool, name='do_augmentation')
        self.loss_class_weight = tf.placeholder(tf.float32, shape=(num_classes, num_classes), name='weights')
        if cfg.db_name == 'honda':
            self.input = tf.placeholder(tf.float32, shape=(batch_size, const.frame_height, const.frame_width,
                                                           const.num_channels), name='context_input')
        else:
            self.input = tf.placeholder(tf.float32, shape=(batch_size, const.max_frame_size, const.max_frame_size,
                                                           const.num_channels), name='context_input')

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
                                   lambda: nn_utils.augment(self.input,cfg.preprocess_func, horizontal_flip=True, vertical_flip=False,
                                                            rotate=0, crop_probability=0, color_aug_probability=0)
                                   , lambda: nn_utils.center_crop(self.input,cfg.preprocess_func))



        with slim.arg_scope(inception_v1_arg_scope()):
            # _, train_end_points = inception_v1(aug_imgs, num_classes,
            #                                    dropout_keep_prob=dropout_keep_prob,
            #                                    is_training=True,reuse=reuse, scope=scope)

            _, self.val_end_points = inception_v1(aug_imgs, num_classes,
                                             dropout_keep_prob=dropout_keep_prob,
                                             is_training=False,reuse=None, scope=scope)


        def  cal_metrics(end_points):
            gt = tf.argmax(self.gt_lbls, 1);
            logits = tf.reshape(end_points['Logits'], [-1, num_classes])
            pre_logits = end_points['Mixed_5c']

            center_supervised_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.gt_lbls,
                                                                                         logits=logits,
                                                                                         name='xentropy_center')
            loss = tf.reduce_mean(center_supervised_cross_entropy, name='xentropy_mean')
            predictions = tf.reshape(end_points['Predictions'], [-1, num_classes])
            class_prediction = tf.argmax(predictions, 1)
            supervised_correct_prediction = tf.equal(gt, class_prediction)
            supervised_correct_prediction_cast = tf.cast(supervised_correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(supervised_correct_prediction_cast)
            confusion_mat = tf.confusion_matrix(gt, class_prediction, num_classes=num_classes)
            _, accumulated_accuracy = tf.metrics.accuracy(gt, class_prediction)
            _, per_class_acc_acc = tf.metrics.mean_per_class_accuracy(gt, class_prediction,num_classes=num_classes)

            per_class_acc_acc = tf.reduce_mean(per_class_acc_acc)
            class_prediction = tf.nn.softmax(logits)
            return loss,pre_logits,accuracy,confusion_mat,accumulated_accuracy,per_class_acc_acc,class_prediction,logits

        # self.train_loss,self.train_pre_logits,self.train_accuracy,self.train_confusion_mat,\
        #                 self.train_accumulated_accuracy,self.train_per_class_acc_acc ,self.train_class_prediction = cal_metrics(train_end_points);


        self.val_loss,self.val_pre_logits,self.val_accuracy, self.val_confusion_mat,\
                        self.val_accumulated_accuracy,self.val_per_class_acc_acc ,self.val_class_prediction,self.val_logits = cal_metrics(self.val_end_points);