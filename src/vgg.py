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
"""
Usage:
    with slim.arg_scope(vgg.vgg_arg_scope()):
        outputs, end_points = vgg.vgg_16(inputs)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def vgg_arg_scope(weight_decay=0.000001):
    """Defines the VGG arg scope.
    Args:
        weight_decay: The l2 regularization coefficient.
    Returns:
        An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                         activation_fn=tf.nn.relu,
                         weights_regularizer=slim.l2_regularizer(weight_decay),
                         biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

class Net(object):
    def __init__(self, net_input, mission, net_name, stack,
                 build_pyramid=True, build_pyramid_layers=4,
                 output_stride=None, is_training=True,
                 include_root_block=True, dropout_keep_prob=0.5,
                 net_head='default', fusion='sum',
                 spatial_squeeze=True, fc_conv_padding='VALID', global_pool=False,
                 deformable=None, attention_option=None, reuse=None):

        self.net_input = net_input
        self.mission = mission
        self.net_name = net_name
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze
        self.fc_conv_padding = fc_conv_padding
        self.global_pool = global_pool
        self.net_arg_scope = vgg_arg_scope()
        self.scope = mission + '/' + net_name
        self.predict_layers = []
        with slim.arg_scope(self.net_arg_scope):
            if net_name == 'vgg_16':
                self.vgg_16()
            elif net_name == 'vgg_4':
                self.vgg_4()
            else:
                raise Exception('there is no net called %s' % net_name)

    def vgg_16(self):
        """Oxford Net VGG 16-Layers version D Example.
        Note: All the fully_connected layers have been transformed to conv2d layers.
                To use in classification mode, resize input to 224x224.
        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            num_classes: number of predicted classes. If 0 or None, the logits layer is
                omitted and the input features to the logits layer are returned instead.
            is_training: whether or not the model is being trained.
            dropout_keep_prob: the probability that activations are kept in the dropout
                layers during training.
            spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
            scope: Optional scope for the variables.
            fc_conv_padding: the type of padding to use for the fully connected layer
                that is implemented as a convolutional layer. Use 'SAME' padding if you
                are applying the network in a fully convolutional manner and want to
                get a prediction map downsampled by a factor of 32 as an output.
                Otherwise, the output prediction map will be (input / 32) - 6 in case of
                'VALID' padding.
            global_pool: Optional boolean flag. If True, the input to the classification
                layer is avgpooled to size 1x1, for any input size. (This is not part
                of the original VGG architecture.)
        Returns:
            net: the output of the logits layer (if num_classes is a non-zero integer),
                or the input to the logits layer (if num_classes is 0 or None).
            end_points: a dict of tensors with intermediate activations.
        """
        with tf.variable_scope(self.scope, 'vgg_16', [self.net_input]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                 outputs_collections=end_points_collection):
                net = tf.image.resize_bicubic(self.net_input, [36, 60], name='resize')
                net = slim.repeat(net, 1, slim.conv2d, 64, [5, 5], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], stride=1, scope='pool1')
                net = slim.repeat(net, 1, slim.conv2d, 128, [5, 5], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], stride=1, scope='pool2')
                net = slim.repeat(net, 1, slim.conv2d, 256, [5, 5], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 1, slim.conv2d, 512, [5, 5], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                #net = slim.repeat(net, 1, slim.conv2d, 512, [3, 3], scope='conv5')
                #net = slim.max_pool2d(net, [2, 2], scope='pool5')

                
                # Use conv2d instead of fully_connected layers.
                shape = net.get_shape().as_list()[1:3]
                net = slim.conv2d(net, 512, shape, padding=self.fc_conv_padding, scope='fc6')
                net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training,
                                   scope='dropout6')
                #net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if self.global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    self.end_points['global_pool'] = net
                #net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training, scope='dropout7')
                net = slim.conv2d(net, 2, [1, 1], activation_fn=None, scope='fc8')
                if self.spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                self.end_points[self.scope + '/logits'] = {'P4':net}
                self.predict_layers.append(self.scope + '/P4')
    
    def vgg_4(self):
        with tf.variable_scope(self.scope, 'vgg_4', [self.net_input]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                 outputs_collections=end_points_collection):
                net = tf.image.resize_bicubic(self.net_input, [36, 60], name='resize')
                net = slim.conv2d(net, 20, [5, 5], stride=1, padding='VALID', scope='conv1')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')
                net = slim.conv2d(net, 50, [5, 5], stride=1, padding='VALID', scope='conv2')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')

                # Use conv2d instead of fully_connected layers.
                shape = net.get_shape().as_list()[1:3]
                net = slim.conv2d(net, 500, shape, padding=self.fc_conv_padding, scope='fc3')
                # Convert end_points_collection into a end_point dict.
                self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if self.global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    self.end_points['global_pool'] = net
                net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training, scope='dropout3')
                net = slim.conv2d(net, 2, [1, 1], activation_fn=None, scope='fc4')
                if self.spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc4/squeezed')
                self.end_points[self.scope + '/logits'] = {'P4':net}
                self.predict_layers.append(self.scope + '/P4')
