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
"""Contains the definition of the Inception Resnet V2 architecture.

As described in http://arxiv.org/abs/1602.07261.

    Inception-v4, Inception-ResNet and the Impact of Residual Connections
        on Learning
    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim

class Net(object):
    def __init__(self, net_input, mission, net_name, num_classes,
                 build_pyramid=False, build_pyramid_layers=3,
                 align_feature_maps=True,
                 output_stride=16, is_training=True,
                 activation_fn=tf.nn.relu, dropout_keep_prob=0.8,
                 net_head='default', fusion='sum',
                 deformable=None, attention_option=None, reuse=None):

        self.net_input = net_input
        self.mission = mission
        self.net_name = net_name
        self.build_pyramid = build_pyramid
        self.build_pyramid_layers = build_pyramid_layers
        self.is_training = is_training
        self.output_stride = output_stride
        self.align_feature_maps = align_feature_maps
        self.activation_fn = activation_fn
        self.dropout_keep_prob = dropout_keep_prob
        self.net_head = net_head
        self.fusion = fusion
        self.deformable = deformable
        self.attention_option = attention_option
        self.num_classes = num_classes
        self.net_arg_scope = self.inception_resnet_v2_arg_scope()
        self.scope = mission + '/' + net_name
        self.reuse = reuse
        self.end_points = {}
        self.predict_layers = []

        with slim.arg_scope(self.net_arg_scope):
            self.inception_resnet_v2()
            self.visual_layer_name()

    def block35(self, net, activation_fn, scale=1.0, scope='Block35'):
        """Builds the 35x35 resnet block."""
        with tf.variable_scope(scope, [net], reuse=self.reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
            mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
            up = slim.conv2d(
                mixed, 
                net.get_shape()[3], 
                1, 
                normalizer_fn=None, 
                activation_fn=None, 
                scope='Conv2d_1x1'
            )
            scaled_up = up * scale
            if activation_fn == tf.nn.relu6:
                # Use clip_by_value to simulate bandpass activation.
                scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

            net += scaled_up
            if activation_fn:
                net = activation_fn(net)
        return net

    def block17(self, net, activation_fn, scale=1.0, scope='Block17'):
        """Builds the 17x17 resnet block."""
        with tf.variable_scope(scope, [net], reuse=self.reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7], scope='Conv2d_0b_1x7')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1], scope='Conv2d_0c_7x1')
            mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
            up = slim.conv2d(
                mixed, 
                net.get_shape()[3], 
                1, 
                normalizer_fn=None, 
                activation_fn=None, 
                scope='Conv2d_1x1'
            )
            scaled_up = up * scale
            if activation_fn == tf.nn.relu6:
                # Use clip_by_value to simulate bandpass activation.
                scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

            net += scaled_up
            if activation_fn:
                net = activation_fn(net)
        return net

    def block8(self, net, activation_fn, scale=1.0, scope='Block8'):
        """Builds the 8x8 resnet block."""
        with tf.variable_scope(scope, [net], reuse=self.reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3], scope='Conv2d_0b_1x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1], scope='Conv2d_0c_3x1')
            mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
            up = slim.conv2d(
                mixed, 
                net.get_shape()[3], 
                1, 
                normalizer_fn=None, 
                activation_fn=None, 
                scope='Conv2d_1x1'
            )
            scaled_up = up * scale
            if activation_fn == tf.nn.relu6:
                # Use clip_by_value to simulate bandpass activation.
                scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

            net += scaled_up
            if activation_fn:
                net = activation_fn(net)
        return net

    def add_point(self, point, net):
        self.end_points[self.scope + '/' + point] = net

    def inception_resnet_v2(self):
        """Inception model from  http://arxiv.org/abs/1602.07261.

        Constructs an Inception Resnet v2 network from inputs to the given final
        endpoint. This method can construct the network up to the final inception
        block Conv2d_7b_1x1.

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            final_endpoint: specifies the endpoint to construct the network up to. It
                can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
                'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
                'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
            output_stride: A scalar that specifies the requested ratio of input to
                output spatial resolution. Only supports 8 and 16.
            align_feature_maps: When true, changes all the VALID paddings in the network
                to SAME padding so that the feature maps are aligned.
            scope: Optional variable_scope.
            activation_fn: Activation function for block scopes.

        Returns:
            tensor_out: output tensor corresponding to the final_endpoint.
            end_points: a set of activations for external use, for example summaries or
                losses.

        Raises:
            ValueError: if final_endpoint is not set to one of the predefined values,
                or if the output_stride is not 8 or 16, or if the output_stride is 8 and
                we request an end point after 'PreAuxLogits'.
        """
        if self.output_stride != 16:
            raise ValueError('output_stride must be 16.')

        padding = 'SAME' if self.align_feature_maps else 'VALID'

        with tf.variable_scope(self.scope, [self.net_input], reuse=self.reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                    # 144 x 240 x 3
                    net = self.net_input
                    # 72 x 120 x 32
                    net = slim.conv2d(net, 32, 3, stride=2, padding=padding, scope='Conv2d_1a_3x3')
                    self.add_point('Conv2d_1a_3x3', net)
                    # 72 x 120 x 32
                    net = slim.conv2d(net, 32, 3, padding=padding, scope='Conv2d_2a_3x3')
                    self.add_point('Conv2d_2a_3x3', net)
                    # 72 x 120 x 64
                    net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                    self.add_point('Conv2d_2b_3x3', net)
                    # 36 x 60 x 64
                    net = slim.max_pool2d(net, 3, stride=2, padding=padding, scope='MaxPool_3a_3x3')
                    self.add_point('MaxPool_3a_3x3', net)
                    # 36 x 60 x 80
                    net = slim.conv2d(net, 80, 1, padding=padding, scope='Conv2d_3b_1x1')
                    self.add_point('Conv2d_3b_1x1', net)
                    # 36 x 60 x 192
                    net = slim.conv2d(net, 192, 3, padding=padding, scope='Conv2d_4a_3x3')
                    self.add_point('Conv2d_4a_3x3', net)
                    # 18 x 30 x 192
                    net = slim.max_pool2d(net, 3, stride=2, padding=padding, scope='MaxPool_5a_3x3')
                    self.add_point('MaxPool_5a_3x3', net)

                    # 18 x 30 x 320
                    with tf.variable_scope('Mixed_5b'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5, scope='Conv2d_0b_5x5')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3, scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3, scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME', scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(tower_pool, 64, 1, scope='Conv2d_0b_1x1')
                        net = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

                    net = slim.repeat(net, 10, self.block35, scale=0.17, activation_fn=self.activation_fn)

                    self.add_point('Mixed_5b', net)
                
                    #9 x 15 x 1088
                    with tf.variable_scope('Mixed_6a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(
                                net, 
                                384, 
                                3, 
                                stride=2, 
                                padding=padding, scope='Conv2d_1a_3x3'
                            )
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3, scope='Conv2d_0b_3x3')
                            tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3, 
                                stride=2, padding=padding, scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_pool = slim.max_pool2d(net, 3, 
                                stride=2, padding=padding, scope='MaxPool_1a_3x3')
                        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

                    with slim.arg_scope([slim.conv2d], rate=1):
                        net = slim.repeat(net, 20, self.block17, scale=0.10, activation_fn=self.activation_fn)

                    self.add_point('Mixed_6a', net)

                    # 5 x 8 x 2080
                    with tf.variable_scope('Mixed_7a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, 
                                stride=2, padding=padding, scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, 
                                stride=2, padding=padding, scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3, scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, 
                                stride=2, padding=padding, scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.max_pool2d(net, 3, 
                                stride=2, padding=padding, scope='MaxPool_1a_3x3')
                        net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

                    net = slim.repeat(net, 9, self.block8, scale=0.20, activation_fn=self.activation_fn)
                    net = self.block8(net, activation_fn=None)

                    self.add_point('Mixed_7a', net)

                    self._build_pyramid()
                    self._Features_To_Predictions()

    def _build_pyramid(self):

        with tf.variable_scope('get_feature_maps'):
            # 5 x 8 x 2080
            self.end_points[self.scope + '/C5'] = self.end_points[self.scope + '/Mixed_7a']

        with tf.variable_scope('feature_pyramid'):
            # 5 x 8 x 1024
            self.end_points[self.scope + '/P5'] = slim.conv2d(
                self.end_points[self.scope + '/C5'],
                num_outputs=1024,
                kernel_size=[1, 1], 
                stride=1, 
                scope='build_P5'
            )
            self.predict_layers.append(self.scope + '/P5')

        if not self.build_pyramid or self.build_pyramid_layers <= 1:
            return

        with tf.variable_scope('get_feature_maps'):
            # 9 x 15 x 1088
            self.end_points[self.scope + '/C4'] = self.end_points[self.scope + '/Mixed_6a']

        with tf.variable_scope('feature_pyramid'):
            # 9 x 15 x 256
            p = self.end_points[self.scope + '/P5']
            c = self.end_points[self.scope + '/C4']
            up_shape = tf.shape(c)
            up = tf.image.resize_bicubic(p, [up_shape[1], up_shape[2]], name='build_P4/up_sample')
            c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1, scope='build_P4/reduce_dimension')
            p = slim.conv2d(up, num_outputs=256, kernel_size=[1, 1], stride=1, scope='build_P4/reduce_same')
            p = p + c
            p = slim.conv2d(p, num_outputs=256, kernel_size=[3, 3], stride=1, padding='SAME', scope='build_P4')
            self.end_points[self.scope + '/P4'] = p
            self.predict_layers.append(self.scope + '/P4')

        if self.build_pyramid_layers <= 2:
            return

        with tf.variable_scope('get_feature_maps'):
            # 18 x 30 x 320
            self.end_points[self.scope + '/C3'] = self.end_points[self.scope + '/Mixed_5b']

        with tf.variable_scope('feature_pyramid'):
            # 18 x 30 x 64
            p = self.end_points[self.scope + '/P4']
            c = self.end_points[self.scope + '/C3']
            up_shape = tf.shape(c)
            up = tf.image.resize_bicubic(p, [up_shape[1], up_shape[2]], name='build_P3/up_sample')
            c = slim.conv2d(c, num_outputs=64, kernel_size=[1, 1], stride=1, scope='build_P3/reduce_dimension')
            p = slim.conv2d(up, num_outputs=64, kernel_size=[1, 1], stride=1, scope='build_P3/reduce_same')
            p = p + c
            p = slim.conv2d(p, num_outputs=64, kernel_size=[3, 3], stride=1, padding='SAME', scope='build_P3')
            self.end_points[self.scope + '/P3'] = p
            self.predict_layers.append(self.scope + '/P3')

    def _Features_To_Predictions(self):

        with tf.variable_scope('Logits'):
            self.end_points[self.scope + '/concat_global_pool'] = None
            for layer_name in self.predict_layers:
                lay_name = layer_name.split('/')[-1]
                net = self.end_points[layer_name]
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name=lay_name + '_global_pool')
                self.end_points[layer_name + '/gloval_pool'] = net
                # Dropout
                net = slim.dropout(
                    net, 
                    self.dropout_keep_prob, 
                    is_training=self.is_training, 
                    scope=lay_name + '_Dropout'
                )
                self.end_points[layer_name + '/Dropout'] = net
                # Concat
                if self.end_points[self.scope + '/concat_global_pool'] is None:
                    self.end_points[self.scope + '/concat_global_pool'] = net
                else:
                    self.end_points[self.scope + '/concat_global_pool'] = tf.concat(
                        [self.end_points[self.scope + '/concat_global_pool'], net],
                        axis=3
                    )

            self.end_points[self.scope + '/logits'] = slim.conv2d(
                self.end_points[self.scope + '/concat_global_pool'],
                self.num_classes,
                kernel_size=[1, 1],
                padding='VALID',
                activation_fn=None,
                normalizer_fn=None,
                scope='concat_logits'
            )
            self.end_points[self.scope + '/logits'] = tf.squeeze(
                self.end_points[self.scope + '/logits'],
                [1, 2],
                name='SpatialSqueeze'
            )

    def inception_resnet_v2_arg_scope(self,
                                      #weight_decay=0.00004,
                                      weight_decay=0.00000,
                                      batch_norm_decay=0.9997,
                                      batch_norm_epsilon=0.001):
        """Returns the scope with the default parameters for inception_resnet_v2.

        Args:
            weight_decay: the weight decay for weights variables.
            batch_norm_decay: decay for the moving average of batch_norm momentums.
            batch_norm_epsilon: small float added to variance to avoid dividing by zero.
            activation_fn: Activation function for conv2d.

        Returns:
            a arg_scope with the parameters needed for inception_resnet_v2.
        """
        # Set weight_decay for weights in conv2d and fully_connected layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                             weights_regularizer=slim.l2_regularizer(weight_decay),
                             biases_regularizer=slim.l2_regularizer(weight_decay)):

            batch_norm_params = {
                'decay': batch_norm_decay,
                'epsilon': batch_norm_epsilon,
                'fused': None,  # Use fused batch norm if possible.
            }
            # Set activation_fn and parameters for batch_norm.
            with slim.arg_scope([slim.conv2d], activation_fn=self.activation_fn,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params) as scope:
                return scope

    def visual_layer_name(self):
        self.visual_layer_name = [
            self.scope + '/Conv2d_1a_3x3',
            self.scope + '/Conv2d_2a_3x3',
            self.scope + '/Conv2d_2b_3x3',
            self.scope + '/Conv2d_3b_1x1',
            self.scope + '/Conv2d_4a_3x3',
            self.scope + '/Mixed_5b',
            self.scope + '/Mixed_6a',
            self.scope + '/Mixed_7a',
            self.scope + '/C3',
            self.scope + '/C4',
            self.scope + '/C4',
            self.scope + '/P3',
            self.scope + '/P4',
            self.scope + '/P5',
        ]
