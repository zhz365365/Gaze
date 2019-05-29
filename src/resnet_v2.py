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

import math
import tensorflow as tf
import deformation as df

import resnet_utils

slim = tf.contrib.slim

class Net(object):
    def __init__(self, net_input, mission, net_name, stack,
                 block1=3, block2=4, block3=6, block4=3, feature_channels=512,
                 build_pyramid=True, build_pyramid_layers=4,
                 output_stride=None, is_training=True,
                 include_root_block=True, dropout_keep_prob=0.8,
                 net_head='default', fusion='sum',
                 deformable=None, attention_option=None, reuse=None):

        self.net_input = net_input
        self.mission = mission
        self.net_name = net_name
        self.build_pyramid = build_pyramid
        self.build_pyramid_layers = stack
        self.is_training = is_training
        self.output_stride = output_stride
        self.include_root_block = include_root_block
        self.dropout_keep_prob = dropout_keep_prob
        self.net_head = net_head
        self.fusion=fusion
        self.deformable = deformable
        self.attention_option = attention_option
        self.net_arg_scope = resnet_utils.resnet_arg_scope()
        self.scope = mission + '/' + net_name
        self.reuse = reuse
        self.end_points = {}
        self.predict_layers = []
        self.block_num = [block1, block2, block3, block4]
        self.feature_channels=feature_channels

        with slim.arg_scope(self.net_arg_scope):
            if net_name == 'resnet_v2':
                self.resnet_v2_net()
            elif net_name == 'resnext_v2':
                self.resnext_v2_net()
            else:
                raise Exception('there is no net called %s' % net_name)
            self.resnet_v2()

    @slim.add_arg_scope
    def bottleneck(self, inputs, depth, depth_bottleneck, stride, rate=1, 
                   deformable=None, attention_option=None,
                   outputs_collections=None, scope=None):
        """Bottleneck residual unit variant with BN before convolutions.
    
        This is the full preactivation residual unit variant proposed in [2]. See
        lim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')ig. 1(b) of [2] for its definition. Note that we use here the bottleneck
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
            # preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
            preact = tf.nn.relu(inputs)
            if depth == depth_in:
                shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                       normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

            residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
            if stride == 1:
                # Deformable blocks
                if deformable is not None and deformable == '1':
                    end_point = 'Deformation'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Deform'):
                            residual_feature = slim.conv2d(residual, depth_bottleneck, 3, stride, rate=rate, padding='SAME', scope='feature')
                            residual_shape = residual_feature.get_shape().as_list()
                            b = residual_shape[0]; h = residual_shape[1]; w = residual_shape[2]; c = residual_shape[3];
                            residual_offset = slim.conv2d(inputs, 2*depth_bottleneck, 3, stride, rate=rate, padding='SAME', scope='offset')
                            residual = df._to_b_h_w_c(df.tf_batch_map_offsets(df._to_bc_h_w(residual_feature, residual_shape), 
                                                                              df._to_bc_h_w_2(residual_offset, residual_shape)), residual_shape)
                else:
                    residual = slim.conv2d(residual, depth_bottleneck, 3, stride, rate=rate, padding='SAME', scope='conv2')
                # Attention blocks
                if attention_option is not None and attention_option[0] == '1': 
                    end_point = 'Attention_S'
                    with tf.variable_scope(end_point):
                        residual_shape = residual.get_shape().as_list()
                        b = residual_shape[0]; h = residual_shape[1]; w = residual_shape[2]; c = residual_shape[3];
                        with tf.variable_scope('Spatial'):
                            attention_map = slim.conv2d(residual, c, 3, stride=1, rate=rate, scope='attention_s_kernel')
                            attention_map = df._to_b_h_w_c(tf.nn.softmax(df._to_bc_hw(attention_map, residual_shape)), residual_shape)
                        residual = residual * attention_map
            
                if attention_option is not None and attention_option[1] == '1':
                    end_point = 'Attention_C'
                    with tf.variable_scope(end_point):
                        residual_shape = residual.get_shape().as_list()
                        b = residual_shape[0]; h = residual_shape[1]; w = residual_shape[2]; c = residual_shape[3];
                        with tf.variable_scope('Channel'):
                            attention_map = slim.conv2d(residual, c, 3, stride=1, rate=rate, scope='attention_c_kernel')
                            attention_map = tf.nn.softmax(tf.reduce_mean(tf.reshape(attention_map, [b*h*w, c]), axis=0))
                        residual = residual * attention_map

                if attention_option is not None and attention_option[2] == '1': 
                    end_point = 'Attention_S'
                    with tf.variable_scope(end_point):
                        residual_shape = residual.get_shape().as_list()
                        b = residual_shape[0]; h = residual_shape[1]; w = residual_shape[2]; c = residual_shape[3];
                        with tf.variable_scope('Spatial'):
                            attention_map = slim.conv2d(residual, c, 3, stride=1, rate=rate, scope='attention_s_kernel')
                            attention_map = df._to_b_h_w_c(tf.nn.softmax(df._to_bc_hw(attention_map, residual_shape)), residual_shape)
                        residual = residual * attention_map
            
                if attention_option is not None and attention_option[3] == '1': 
                    end_point = 'Attention_M'
                    with tf.variable_scope(end_point):
                        residual_shape = residual.get_shape().as_list()
                        b = residual_shape[0]; h = residual_shape[1]; w = residual_shape[2]; c = residual_shape[3];
                        with tf.variable_scope('Modulation'):
                            attention_map = slim.conv2d(residual, c, 3, stride=1, rate=rate, scope='attention_m_kernel')
                            attention_map = tf.clip_by_value(attention_map, 0, 1)
                        residual = residual * attention_map

            else:
                residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')

            residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='conv3')

            output = shortcut + residual

            return slim.utils.collect_named_outputs(outputs_collections,
                                                    sc.name,
                                                    output)

    @slim.add_arg_scope
    def bottle_x_neck(self, inputs, depth, depth_bottleneck, stride, rate=1, 
                      deformable=None, attention_option=None,
                      outputs_collections=None, scope=None):
        """
        ResNext
        """
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            # preact = slim.batch_norm(inputs, scope='preact')
            preact = tf.nn.relu(net)
            if depth == depth_in:
                shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                       normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')
            """
            # ResNet
            residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
            residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
            residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='conv3')
            """
            depth_bottleneck_per = depth_bottleneck / 32
            residual_split = []
            for i in range(32):
                net = slim.conv2d(preact, depth_bottleneck_per, [1, 1], stride=1, scope='conv1_%d' % i)
                net = resnet_utils.conv2d_same(net, depth_bottleneck_per, 3, stride, rate=rate, scope='conv2_%d' % i)
                residual_split.append(net)
            residual = tf.concat(residual_split, axis=3)
            residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='conv3')

            output = shortcut + residual

            return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

    def resnet_v2(self):
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
        with tf.variable_scope(self.scope, 'resnet_v2', [self.net_input], reuse=self.reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d,
                                self.bottleneck if(self.net_name.startswith('resnet')) else self.bottle_x_neck,
                                resnet_utils.stack_blocks_dense],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                    net = self.net_input
                    if self.include_root_block:
                        if self.output_stride is not None:
                            if self.output_stride % 4 != 0:
                                raise ValueError('The output_stride needs to be a multiple of 4.')
                            self.output_stride /= 4
                        # We do not include batch normalization or activation functions in
                        # conv1 because the first ResNet unit will perform these. Cf.
                        # Appendix of [2].
                        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                            if self.net_head == 'default':
                                net = resnet_utils.conv2d_same(net, self.feature_channels/4/(2**3), 7, stride=2, scope='conv1')
                                net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool1')
                            elif self.net_head == 'stem':
                                net = resnet_utils.conv2d_same(net, self.feature_channels/4/(2**3), 3, stride=2, scope='conv1')
                                net = resnet_utils.conv2d_same(net, self.feature_channels/4/(2**3), 1, stride=1, scope='conv2')
                                net = resnet_utils.conv2d_same(net, self.feature_channels/4/(2**3), 1, stride=1, scope='conv3')
                                net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
                            else:
                                raise Exception('wrong net head!')
                    net = resnet_utils.stack_blocks_dense(net, self.blocks)
                    # Convert end_points_collection into a dictionary of end_points.
                    self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                    self._build_pyramid()
                    self._Featuremaps_To_Predictions()

    def _Featuremaps_To_Predictions(self):
        self.end_points[self.scope + '/logits'] = {}
        for layer_name in self.predict_layers:
            lay_name = layer_name.split('/')[-1]
            net = self.end_points[layer_name]
            shape = net.get_shape().as_list()[1:3]
            stride_down = max(math.ceil(shape[0]/5), math.ceil(shape[1]/8))
            if stride_down != 1:
                net = slim.max_pool2d(net, [5, 8], stride=stride_down, padding='SAME', scope=lay_name+'_downsample_to_5x8x%d'%self.feature_channels)
                self.end_points[layer_name + '/downsample'] = net
            shape = net.get_shape().as_list()[1:3]
            net = slim.conv2d(net, self.feature_channels, shape, stride=1, padding='VALID', scope=lay_name + '_to_1x1x%d_1'%self.feature_channels)
            net = slim.dropout(net, self.dropout_keep_prob, scope=lay_name + '_dropout')
            net = slim.conv2d(net, 2, [1, 1], padding='VALID', activation_fn=None, normalizer_fn=None, scope=lay_name + '_logits')
            net = tf.squeeze(net, [1, 2], name=lay_name + '_spatialsqueeze')
            self.end_points[self.scope + '/logits'][lay_name] = net

    def _build_pyramid(self):

        def _self_attention(U):
            shape = tf.shape(U)
            U = tf.transpose(U, [0, 3, 1, 2])
            U = tf.reshape(U, [-1, shape[1]*shape[2]])
            U = tf.nn.softmax(U)
            U = tf.reshape(U, [-1, shape[3], shape[1], shape[2]])
            U = tf.transpose(U, [0, 2, 3, 1])
            return U

        def _repadding_Up_to_C(Up, C):
            shape_Up = Up.get_shape().as_list()
            shape_C = C.get_shape().as_list()
            kernel_shape = [shape_Up[1]-shape_C[1]+1, shape_Up[2]-shape_C[2]+1]
            return kernel_shape

        with tf.variable_scope('get_feature_maps'):
            self.end_points[self.scope + '/C' + str(4)] = self.end_points[self.scope + '/block' + str(4)]
            for i in range(3, 0, -1):
                self.end_points[self.scope + '/C' + str(i)] = self.end_points[self.scope + '/block' + str(i) + '/unit_' + str(self.block_num[i-1]-1) + '/bottleneck_v2']

        with tf.variable_scope('feature_pyramid'):
            """
            net = self.end_points[self.scope + '/C5']
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='P5_postnorm')
            self.end_points[self.scope + '/P5'] = net
            # self.end_points[self.scope + '/P5_visual'] = df._to_c_h_w_1(tf.image.resize_bicubic(net, [36, 60], name='build_P%d/visual' % 5))
            self.predict_layers.append(self.scope + '/P5')

            for layer in range(4, 5-self.build_pyramid_layers, -1):
                P = self.end_points[self.scope + '/P' + str(layer + 1)]
                C = self.end_points[self.scope + '/C' + str(layer)]
                C = resnet_utils.stack_blocks_dense(
                    C,
                    self.reduce_dimension('build_P%d' % layer)
                )
                C = slim.batch_norm(C, activation_fn=tf.nn.relu, scope='build_P%d/reduce_dimension_postnorm' % layer)
                # resize_image only can be operated by CPU
                # Up_sample_shape = tf.shape(C)
                # Up_sample = tf.image.resize_bicubic(P, [Up_sample_shape[1], Up_sample_shape[2]], name='build_P%d/up_sample_bicubic' % layer)
                # conv2d_transpose
                Up_sample = slim.conv2d_transpose(P, P.get_shape().as_list()[3], [2, 2], stride=2, padding='SAME', scope='build_P%d/up_sample_conv2d' % layer) 
                Up_sample = slim.conv2d(Up_sample, P.get_shape().as_list()[3], _repadding_Up_to_C(Up_sample, C), stride=1, padding='VALID', scope='build_P%d/up_sample_pad' % layer)
                if self.fusion == 'sum':
                    # element_wise_sum
                    P = Up_sample + C
                elif self.fusion == 'concat':
                    # concat
                    P = tf.concat([Up_sample, C], axis=2)
                elif self.fusion == 'attention':
                    # self-attention & element_wise_dot
                    Up_sample = _self_attention(Up_sample)
                    P = Up_sample * C
                else:
                    raise Exception('wrong fusion method!')
                P = resnet_utils.stack_blocks_dense(
                    P,
                    self.reduce_confusion('build_P%d' % layer)
                )
                P = slim.batch_norm(P, activation_fn=tf.nn.relu, scope='build_P%d/reduce_confusion_postnorm' % layer)
                self.end_points[self.scope + '/P' + str(layer)] = P
                # self.end_points[self.scope + '/P' + str(layer) + '_visual'] = df._to_c_h_w_1(tf.image.resize_bicubic(P, [36, 60], name='build_P%d/visual' % layer))
                self.predict_layers.append(self.scope + '/P' + str(layer))
            """
            for layer in range(4, 4-self.build_pyramid_layers, -1):
                net = self.end_points[self.scope + '/C' + str(layer)]
                net = resnet_utils.stack_blocks_dense(net, self.reduce_dimension('build_P%d' % layer))
                # net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='P%d_postnorm' % layer)
                net = tf.nn.relu(net)
                self.end_points[self.scope + '/P' + str(layer)] = net
                # self.end_points[self.scope + '/P' + str(layer) + '_visual'] = df._to_c_h_w_1(tf.image.resize_bicubic(net, [36, 60], name='build_P%d/visual' % layer))
                self.predict_layers.append(self.scope + '/P' + str(layer))

    def resnet_v2_block(self, scope, base_depth, num_units, stride):
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

        if 'block4' in scope:
            return resnet_utils.Block(scope, self.bottleneck, [{
                'depth': base_depth * 4,
                'depth_bottleneck': base_depth,
                'stride': 1,
                'rate': 1,
                'deformable': self.deformable,
                'attention_option': self.attention_option
            }] * num_units)
        else:
            return resnet_utils.Block(scope, self.bottleneck, [{
                'depth': base_depth * 4,
                'depth_bottleneck': base_depth,
                'stride': 1
            }] * (num_units - 1) + [{
                'depth': base_depth * 4,
                'depth_bottleneck': base_depth,
                'stride': stride
            }])

    def resnext_v2_block(self, scope, base_depth, num_units, stride):
        return resnet_utils.Block(scope, self.bottle_x_neck, [{
            'depth': base_depth * 2,
            'depth_bottleneck': base_depth,
            'stride' : 1
        }] * (num_units - 1) + [{
            'depth': base_depth * 2,
            'depth_bottleneck' : base_depth,
            'stride' : stride
        }])

    def resnet_v2_net(self):
        self.blocks = [
            self.resnet_v2_block('block1', base_depth=self.feature_channels/4/(2**3), num_units=self.block_num[0], stride=2),
            self.resnet_v2_block('block2', base_depth=self.feature_channels/4/(2**2), num_units=self.block_num[1], stride=2),
            self.resnet_v2_block('block3', base_depth=self.feature_channels/4/(2**1), num_units=self.block_num[2], stride=2),
            self.resnet_v2_block('block4', base_depth=self.feature_channels/4/(2**0), num_units=self.block_num[3], stride=1),
        ]
    def resnext_v2_net(self):
        self.blocks = [
            self.resnext_v2_block('block1', base_depth=self.feature_channels/2/(2**3), num_units=self.block_num[0], stride=2),
            self.resnext_v2_block('block2', base_depth=self.feature_channels/2/(2**2), num_units=self.block_num[1], stride=2),
            self.resnext_v2_block('block3', base_depth=self.feature_channels/2/(2**1), num_units=self.block_num[2], stride=2),
            self.resnext_v2_block('block4', base_depth=self.feature_channels/2/(2**0), num_units=self.block_num[3], stride=1),
        ]
    def reduce_dimension(self, pyramid_layer):
        if self.net_name.startswith('resnet'):
            blocks = [
                self.resnet_v2_block(pyramid_layer + '/reduce_dimension', base_depth=self.feature_channels/4, num_units=1, stride=1),
            ]
        elif self.net_name.startswith('resnext'):
            blocks = [
                self.resnext_v2_block(pyramid_layer + '/reduce_dimension', base_depth=self.feature_channels/2, num_units=1, stride=1),
            ]
        else:
            raise Exception('wrong net name!')
        return blocks

    def reduce_confusion(self, pyramid_layer):
        if self.net_name.startswith('resnet'):
            blocks = [
                self.resnet_v2_block(pyramid_layer + '/reduce_confusion', base_depth=self.feature_channels/4, num_units=1, stride=1),
            ]
        elif self.net_name.startswith('resnext'):
            blocks = [
                self.resnext_v2_block(pyramid_layer + '/reduce_confusion', base_depth=self.feature_channels/2, num_units=1, stride=1),
            ]
        else:
            raise Exception('wrong net name!')
        return blocks
