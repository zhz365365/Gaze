# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

import inception_utils
import tensorflow as tf
import deformation as df

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

def inception_v3_base(inputs,
                      final_endpoint='Conv2d_4a_3x3',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
    """Inception model from http://arxiv.org/abs/1512.00567.

    Constructs an Inception v3 network from inputs to the given final endpoint.
    This method can construct the network up to the final inception block
    Mixed_7c.

    Note that the names of the layers in the paper do not correspond to the names
    of the endpoints registered by this function although they build the same
    network.

    Here is a mapping from the old_names to the new names:
    Old name          | New name
    =======================================
    conv0             | Conv2d_1a_3x3
    conv1             | Conv2d_2a_3x3
    conv2             | Conv2d_2b_3x3
    pool1             | MaxPool_3a_3x3
    conv3             | Conv2d_3b_1x1
    conv4             | Conv2d_4a_3x3
    pool2             | MaxPool_5a_3x3
    mixed_35x35x256a  | Mixed_5b
    mixed_35x35x288a  | Mixed_5c
    mixed_35x35x288b  | Mixed_5d
    mixed_17x17x768a  | Mixed_6a
    mixed_17x17x768b  | Mixed_6b
    mixed_17x17x768c  | Mixed_6c
    mixed_17x17x768d  | Mixed_6d
    mixed_17x17x768e  | Mixed_6e
    mixed_8x8x1280a   | Mixed_7a
    mixed_8x8x2048a   | Mixed_7b
    mixed_8x8x2048b   | Mixed_7c

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
            final_endpoint: specifies the endpoint to construct the network up to. It
            can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
            'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
            'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
            'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
        min_depth: Minimum depth value (number of channels) for all convolution ops.
            Enforced when depth_multiplier < 1, and not an active constraint when
            depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        scope: Optional variable_scope.

    Returns:
        tensor_out: output tensor corresponding to the final_endpoint.
        end_points: a set of activations for external use, for example summaries or
            losses.

    Raises:
        ValueError: if final_endpoint is not set to one of the predefined values,
            or depth_multiplier <= 0
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with variable_scope.variable_scope(scope, 'InceptionV3', [inputs]):
        with arg_scope(
            [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
            stride=1,
            padding='VALID'):
            # 299 x 299 x 3
            end_point = 'Conv2d_1a_3x3'
            net = layers.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # 149 x 149 x 32
            end_point = 'Conv2d_2a_3x3'
            net = layers.conv2d(net, depth(32), [3, 3], scope=end_point)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # 147 x 147 x 32
            end_point = 'Conv2d_2b_3x3'
            net = layers.conv2d(
                net, depth(64), [3, 3], padding='SAME', scope=end_point)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # 147 x 147 x 64
            end_point = 'MaxPool_3a_3x3'
            net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # 73 x 73 x 64
            end_point = 'Conv2d_3b_1x1'
            net = layers.conv2d(net, depth(80), [1, 1], scope=end_point)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # 73 x 73 x 80.
            end_point = 'Conv2d_4a_3x3'
            net = layers.conv2d(net, depth(192), [3, 3], scope=end_point)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # 71 x 71 x 192.
    
        raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v3_branch(net,
                        end_points,
                        num_classes,
                        deformable,
                        attention_option,
                        dropout_keep_prob=0.8,
                        final_endpoint='Logits',
                        min_depth=16,
                        depth_multiplier=1.0,
                        spatial_squeeze=True,   
                        scope=None):
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with variable_scope.variable_scope(scope, 'InceptionV3', [net, end_points]):
        with arg_scope(
            [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
            stride=1,
            padding='SAME'):
            
            # Deformable blocks
            if deformable == '1':
                end_point = 'Deformation'
                with variable_scope.variable_scope(end_point):
                    with variable_scope.variable_scope('Deform'):
                        net_feature = end_points['Conv2d_4a_3x3']
                        net_shape = net_feature.get_shape().as_list()
                        b = net_shape[0]; h = net_shape[1]; w = net_shape[2]; c = net_shape[3];
                        net_offset = layers.conv2d(end_points['Conv2d_3b_1x1'], depth(2*c), [3, 3], padding='VALID', scope=end_point + '_offset')
                    net = df._to_b_h_w_c(df.tf_batch_map_offsets(df._to_bc_h_w(net_feature, net_shape), df._to_bc_h_w_2(net_offset, net_shape)), net_shape)
                end_points[scope.name + '/' + end_point + '_offset'] = net_offset
                end_points[scope.name + '/' + end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            end_point = 'MaxPool_5a_3x3'
            net = layers_lib.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope=end_point)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # 35 x 35 x 192.
    
            # Attention blocks
            if attention_option[0] == '1': 
                end_point = 'Attention_S'
                with variable_scope.variable_scope(end_point):
                    map_shape = net.get_shape().as_list()
                    b = map_shape[0]; h = map_shape[1]; w = map_shape[2]; c = map_shape[3];
                    with variable_scope.variable_scope('Spatial'):
                        initial_map = layers.conv2d(
                            net, depth(c), [3, 3], scope='Conv2d_4a_3x3')
                        attention_map = df._to_b_h_w_c(tf.nn.softmax(df._to_bc_hw(initial_map, map_shape)), map_shape)
                    net = net * attention_map
                end_points[scope.name + '/' + end_point + '_map'] = attention_map
                end_points[scope.name + '/' + end_point] = net
            
            if attention_option[1] == '1':
                end_point = 'Attention_C'
                with variable_scope.variable_scope(end_point):
                    map_shape = net.get_shape().as_list()
                    b = map_shape[0]; h = map_shape[1]; w = map_shape[2]; c = map_shape[3];
                    with variable_scope.variable_scope('Channel'):
                        initial_map = layers.conv2d(
                            net, depth(c), [3, 3], scope='Conv2d_4a_3x3')
                        attention_map = tf.nn.softmax(tf.reduce_mean(tf.reshape(initial_map, [b*h*w, c]), axis=0))
                    net = net * attention_map
                end_points[scope.name + '/' + end_point + '_map'] = attention_map
                end_points[scope.name + '/' + end_point] = net

            if attention_option[2] == '1': 
                end_point = 'Attention_S'
                with variable_scope.variable_scope(end_point):
                    map_shape = net.get_shape().as_list()
                    b = map_shape[0]; h = map_shape[1]; w = map_shape[2]; c = map_shape[3];
                    with variable_scope.variable_scope('Spatial'):
                        initial_map = layers.conv2d(
                            net, depth(c), [3, 3], scope='Conv2d_4a_3x3')
                        attention_map = df._to_b_h_w_c(tf.nn.softmax(df.to_bc_hw(initial_map, map_shape)), map_shape)
                    net = net * attention_map
                end_points[scope.name + '/' + end_point + '_map'] = attention_map
                end_points[scope.name + '/' + end_point] = net
            
            if attention_option[3] == '1': 
                end_point = 'Attention_M'
                with variable_scope.variable_scope(end_point):
                    map_shape = net.get_shape().as_list()
                    b = map_shape[0]; h = map_shape[1]; w = map_shape[2]; c = map_shape[3];
                    with variable_scope.variable_scope('Modulation'):
                        initial_map = layers.conv2d(
                            net, depth(c), [3, 3], scope='Conv2d_4a_3x3')
                        attention_map = tf.clip_by_value(initial_map, 0, 1)
                    net = net * attention_map
                end_points[scope.name + '/' + end_point + '_map'] = attention_map
                end_points[scope.name + '/' + end_point] = net
            
            # Inception blocks
            # mixed: 35 x 35 x 256.
            end_point = 'Mixed_5b'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                    branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_1: 35 x 35 x 288.
            end_point = 'Mixed_5c'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                    branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_2: 35 x 35 x 288.
            end_point = 'Mixed_5d'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_3: 17 x 17 x 768.
            end_point = 'Mixed_6a'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net,
                        depth(384), [3, 3],
                        stride=2,
                        padding='VALID',
                        scope='Conv2d_1a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = layers.conv2d(
                        branch_1,
                        depth(96), [3, 3],
                        stride=2,
                        padding='VALID',
                        scope='Conv2d_1a_1x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(
                        net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed4: 17 x 17 x 768.
            end_point = 'Mixed_6b'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = layers.conv2d(
                        branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_5: 17 x 17 x 768.
            end_point = 'Mixed_6c'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # mixed_6: 17 x 17 x 768.
            end_point = 'Mixed_6d'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_7: 17 x 17 x 768.
            end_point = 'Mixed_6e'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = layers.conv2d(
                        branch_0,
                        depth(320), [3, 3],
                        stride=2,
                        padding='VALID',
                        scope='Conv2d_1a_3x3')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = layers.conv2d(
                        branch_1,
                        depth(192), [3, 3],
                        stride=2,
                        padding='VALID',
                        scope='Conv2d_1a_3x3')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(
                        net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = array_ops.concat(
                        [
                            layers.conv2d(
                                branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                            layers.conv2d(
                                branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')
                        ],
                        3)
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = array_ops.concat(
                        [
                            layers.conv2d(
                                branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                            layers.conv2d(
                                branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                        ],
                        3)
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = array_ops.concat(
                        [
                            layers.conv2d(
                                branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                            layers.conv2d(
                                branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')
                        ],
                        3)
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = array_ops.concat(
                        [
                            layers.conv2d(
                                branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                            layers.conv2d(
                                branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                        ],
                        3)
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # Final pooling and prediction
            end_point = 'Logits'
            with variable_scope.variable_scope(end_point):
                kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
                net = layers_lib.avg_pool2d(
                    net,
                    kernel_size,
                    padding='VALID',
                    scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # 1 x 1 x 2048
                net = layers_lib.dropout(
                    net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        
                # 2048
                net = layers.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')

                if spatial_squeeze:
                    net = array_ops.squeeze(net, [1, 2], name='SpatialSqueeze')

            end_points[scope.name + '/' + end_point] = net
            if end_point == final_endpoint:
                return net, end_points
        raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v3(inputs,
                 num_classes,
                 mission,
                 deformable,
                 attention_option,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 reuse=None,
                 create_aux_logits=False,
                 scope='/InceptionV3',
                 global_pool=False):
    """Inception model from http://arxiv.org/abs/1512.00567.

    "Rethinking the Inception Architecture for Computer Vision"

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna.

    With the default arguments this method constructs the exact model defined in
    the paper. However, one can experiment with variations of the inception_v3
    network by changing arguments dropout_keep_prob, min_depth and
    depth_multiplier.

    The default image size used to train this network is 299x299.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether is training or not.
        dropout_keep_prob: the percentage of activation values that are retained.
        min_depth: Minimum depth value (number of channels) for all convolution ops.
            Enforced when depth_multiplier < 1, and not an active constraint when
            depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
            of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
            To use this parameter, the input images must be smaller
            than 300x300 pixels, in which case the output logit layer
            does not contain spatial information and can be removed.
        reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
        scope: Optional variable_scope.

    Returns:
        logits: the pre-softmax activations, a tensor of size
            [batch_size, num_classes]
        end_points: a dictionary from components of the network to the corresponding
            activation.

    Raises:
        ValueError: if 'depth_multiplier' is less than or equal to zero.
    """
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with variable_scope.variable_scope(
        mission + scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with arg_scope(
            [layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
            net, end_points = inception_v3_base(inputs,
                                                scope=scope,
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier)
            net, end_points = inception_v3_branch(net,
                                                  end_points,
                                                  deformable=deformable,
                                                  attention_option=attention_option,
                                                  scope=scope,
                                                  num_classes=num_classes,
                                                  min_depth=min_depth,
                                                  depth_multiplier=depth_multiplier,
                                                  dropout_keep_prob=dropout_keep_prob)

    return net, end_points

inception_v3.default_image_size = 299

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are is large enough.

    Args:
        input_tensor: input tensor of size [batch_size, height, width, channels].
        kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
        a tensor with the kernel size.

    TODO(jrru): Make this function work with unknown shapes. Theoretically, this
    can be done with the code below. Problems are two-fold: (1) If the shape was
    known, it will be lost. (2) inception.tf.contrib.slim.ops._two_element_tuple
    cannot
    handle tensors that define the kernel size.
        shape = tf.shape(input_tensor)
        return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                           tf.minimum(shape[2], kernel_size[1])])

    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
            min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
        ]
    return kernel_size_out

def inception_v3_layer_name():
    visual_layer_name = []
    return visual_layer_name

def _to_bc_h_w_2(x, x_shape):
    #x_shape = [b, h, w, c]
    #[b, h, w, 2c] -> [b*c, h, w, 2]
    return tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1, x_shape[1], x_shape[2], 2])

def _to_bc_h_w(x, x_shape):
    #x_shape = [b, h, w, c]
    #[b, h, w, c] -> [b*c, h, w]
    return tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1, x_shape[1], x_shape[2]])

def _to_bc_hw(x, x_shape):
    #x_shape = [b, h, w, c]
    #[b, h, w, c] -> [b*c, h*w]
    return  tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1, x_shape[1] * x_shape[2]])

def _to_b_h_w_c(x, x_shape):
    #x_shape = [b, h, w, c]
    #[b*c, h, w] -> [b, h, w, c]
    #[b*c, h*w] -> [b, h, w, c]
    #[b, c, h*w] -> [b, h, w, c]
    return tf.transpose(tf.reshape(x, [-1, x_shape[3], x_shape[1], x_shape[2]]), [0, 2, 3, 1])

def tf_flatten(a):
    return tf.reshape(a, [-1])

def tf_repeat(a, repeats, axis=0):
    assert len(a.get_shape()) == 1
    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a


def tf_repeat_2d(a, repeats):
    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a

def tf_batch_map_offsets(x, offsets, order=1):
    input_shape = tf.shape(x)
    batch_channel_size = input_shape[0]
    height_size = input_shape[1]
    width_size = input_shape[2]

    offsets = tf.reshape(offsets, [batch_channel_size, -1, 2])

    grid = tf.meshgrid(tf.range(height_size), tf.range(width_size), indexing='ij')
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_channel_size)
    coords = offsets + grid
    mapped_vals = tf_batch_map_coordinates(x, coords)
    return mapped_vals

def tf_batch_map_coordinates(x, coords, order=1):
    input_shape = tf.shape(x)
    batch_channel_size = input_shape[0]
    height_size = input_shape[1]
    width_size = input_shape[2]
    h_plus_w_size = tf.shape(coords)[1]

    coords = tf.stack([tf.clip_by_value(coords[:,:,0], 0, tf.cast(height_size, tf.float32) - 1), 
                       tf.clip_by_value(coords[:,:,1], 0, tf.cast(width_size, tf.float32) - 1)], axis=-1)

    coords_lt = tf.cast(tf.floor(coords), tf.int32)
    coords_rb = tf.cast(tf.ceil(coords), tf.int32)
    coords_lb = tf.stack([coords_lt[:,:,0], coords_rb[:,:,1]], axis=-1)
    coords_rt = tf.stack([coords_rb[:,:,0], coords_lt[:,:,1]], axis=-1)

    idx = tf_repeat(tf.range(batch_channel_size), h_plus_w_size)

    def _get_vals_by_coords(feature_map, coords):
        indices = tf.stack([idx, tf_flatten(coords[:,:,0]), tf_flatten(coords[:,:,1])], axis=-1)
        vals = tf.gather_nd(feature_map, indices)
        vals = tf.reshape(vals, (batch_channel_size, h_plus_w_size))
        return vals

    vals_lt = _get_vals_by_coords(x, coords_lt)
    vals_rb = _get_vals_by_coords(x, coords_rb)
    vals_lb = _get_vals_by_coords(x, coords_lb)
    vals_rt = _get_vals_by_coords(x, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, tf.float32)
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:,:,0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:,:,0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:,:,1]

    return mapped_vals

inception_v3_arg_scope = inception_utils.inception_arg_scope
