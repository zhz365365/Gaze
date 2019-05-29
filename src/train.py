import tensorflow as tf
import argparse
import input_dataset
import os
import numpy as np
import math
import time
import random

parser = argparse.ArgumentParser(description='')
parser.add_argument('--GPU', default='7', help='the index of gpu')
parser.add_argument('--eyeball', default='left', help='the choose of eyeball')
parser.add_argument('--mission', default='theta', help='vertical or horizontal mission')
parser.add_argument('--batch_size', default=16, help='the size of examples in per batch')
parser.add_argument('--epoch', default=200, help='the train epoch')
parser.add_argument('--lr_boundaries', default='8,16,24,32,40,48,56', help='the boundaries of learning rate')
parser.add_argument('--lr_values', default='0.0004,0.00004,0.000004,0.0000004,0.00000004,0.000000004,0.0000000004,0.00000000004', help='the values of learning_rate')
parser.add_argument('--data_type', default='LOPO_MPIIGaze', help='the silent or move')
parser.add_argument('--data_dir', default='../dataset/', help='the directory of training data')
parser.add_argument('--server', default='77', help='server')
parser.add_argument('--net_name', default='resnet_v2', help='the name of the network')
parser.add_argument('--image_height', default='144', help='the down scale of image')
parser.add_argument('--image_width', default=240, help='the width of image')
parser.add_argument('--classes_theta', default=90, help='the classes of theta')
parser.add_argument('--classes_phi', default=90, help='the classes of phi')
parser.add_argument('--dropout_keep_prob', default=0.5, help='the probility to keep dropout')
parser.add_argument('--net_head', default='default', help='default or stem block')
parser.add_argument('--fusion', default='sum', help='the method to fuse')
parser.add_argument('--deformable', default='0', help='deform or not')
parser.add_argument('--attention_option', default='0000', help='attention_option:\
|S|C|S|M|\
|1|0|0|0|\
|0|1|0|0|\
|1|1|0|0|\
|0|1|1|0|\
|0|0|0|1|\
|0|0|0|0|')
parser.add_argument('--restore_step', default='0', help='the step used to restore')
parser.add_argument('--trainable', default='0', help='train or not')
parser.add_argument('--label_smoothing_method', default='None', help='the method to smooth label')
parser.add_argument('--weight_decay', default=0.01, help='weight decay')
parser.add_argument('--build_pyramid', default='1', help='whether or not build feature map pyramids')
parser.add_argument('--build_pyramid_layers', default='4', help='layers of feature pyramids')
parser.add_argument('--cross_entropy', default='softmax', help='use the softmax or sigmoid')
parser.add_argument('--leave_one', default='14', help='test one')
parser.add_argument('--reduce_mean', default='0', help='preprocess of image')
args = parser.parse_args()
args.data_dir = args.data_dir + args.data_type
args.dropout_keep_prob = float(args.dropout_keep_prob)
args.weight_decay = float(args.weight_decay)
args.classes_theta = int(args.classes_theta)
args.classes_phi = int(args.classes_phi)
args.classes = {
    'theta' : args.classes_theta,
    'phi' : args.classes_phi,
}
args.reduce_mean = bool(args.reduce_mean == '1')
args.build_pyramid = bool(args.build_pyramid == '1')
args.build_pyramid_layers = int(args.build_pyramid_layers)
args.leave_one = int(args.leave_one)
args.batch_size = int(args.batch_size)
args.epoch = int(args.epoch)
args.image_height = int(args.image_height)
args.image_width = int(args.image_width)
lr_boundaries = []
for key in args.lr_boundaries.split(','): 
    lr_boundaries.append(float(key))
lr_values = []
for key in args.lr_values.split(','):
    lr_values.append(float(key))

if args.net_name.startswith('res'):
    import resnet_v2 as Gaze_Scale_Net
elif args.net_name.startswith('Inception'):
    import inception_resnet_v2 as Gaze_Scale_Net
elif args.net_name.startswith('vgg'):
    import vgg as Gaze_Scale_Net
elif args.net_name.startswith('hour'):
    import hourglasses as Gaze_Scale_Net
else:
    raise Exception('wrong net name!')

summary_dir = '../Log/' + args.server + '/event_log/' + args.eyeball + '/' + args.mission
restore_dir = '../Log/' + args.server + '/check_log/' + args.eyeball + '/' + args.mission
checkpoint_dir = '../Log/' + args.server + '/check_log/' + args.eyeball + '/' + args.mission + '/' + args.eyeball + '_' + args.mission + '_model.ckpt'
draw_dir = '../Log/' + args.server + '/draw_log/' + args.eyeball + '/' + args.mission

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
slim = tf.contrib.slim

def arch_net(image, is_training, dropout_keep_prob):
    Net = Gaze_Scale_Net.Net(
        net_input=image,
        mission=args.mission,
        net_name=args.net_name,
        is_training=is_training,
        dropout_keep_prob=dropout_keep_prob,
        net_head=args.net_head,
        stack=args.build_pyramid_layers,
    )

    for key in Net.end_points.keys():
        if isinstance(Net.end_points[key], dict):
            for sub_key in Net.end_points[key].keys():
                print(key + '/' + sub_key, Net.end_points[key][sub_key].get_shape().as_list())
        else:
            print(key, ' ', Net.end_points[key].get_shape().as_list())
    return Net.end_points[Net.scope + '/logits'], Net.end_points

def g_parameter(mission, checkpoint_exclude_scopes):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    variables_to_train = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(mission + '/' + exclusion):
                excluded = True
                variables_to_train.append(var)
                print(var.op.name)
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore, variables_to_train

def smoothing_label(label, classes):
    smoothing_factor = 0.1
    if args.label_smoothing_method == 'all':
        label = tf.one_hot(label, classes)
        label = (1 - smoothing_factor) * label + smoothing_factor / classes
    elif args.label_smoothing_method == 'nearest_2':
    	label = (1 - smoothing_factor) * tf.one_hot(label, classes) + \
                smoothing_factor * 0.4 * tf.one_hot(tf.clip_by_value(label - 1, 0, classes - 1), classes) + \
                smoothing_factor * 0.1 * tf.one_hot(tf.clip_by_value(label - 2, 0, classes - 1), classes) + \
                smoothing_factor * 0.4 * tf.one_hot(tf.clip_by_value(label + 1, 0, classes - 1), classes) + \
                smoothing_factor * 0.1 * tf.one_hot(tf.clip_by_value(label + 2, 0, classes - 1), classes)
    elif args.label_smoothing_method == 'nearest_3' :
        label = (1 - smoothing_factor) * tf.one_hot(label, classes) + \
                smoothing_factor * 0.42 * tf.one_hot(tf.clip_by_value(label - 1, 0, classes - 1), classes) + \
                smoothing_factor * 0.07 * tf.one_hot(tf.clip_by_value(label - 2, 0, classes - 1), classes) + \
                smoothing_factor * 0.01 * tf.one_hot(tf.clip_by_value(label - 3, 0, classes - 1), classes) + \
                smoothing_factor * 0.42 * tf.one_hot(tf.clip_by_value(label + 1, 0, classes - 1), classes) + \
                smoothing_factor * 0.07 * tf.one_hot(tf.clip_by_value(label + 2, 0, classes - 1), classes) + \
                smoothing_factor * 0.01 * tf.one_hot(tf.clip_by_value(label + 3, 0, classes - 1), classes)
    else:
        label = tf.one_hot(label, classes)
    return label

def loss_function(labels, logits):
    if args.cross_entropy == 'softmax':
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    elif args.cross_entropy == 'sigmoid':
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    else:
        raise Exception('wrong cross entropy loss!')

def get_topk_mean(v, in_):
    b = np.size(v, 0)
    layers = np.size(v, 1)
    ans = np.zeros((b), dtype=np.float32)
    for i in range(b):
        num_v = 1
        ans[i] = in_[i][0]
        for j in range(1, layers, 1):
            if v[i][j] > 0.7 * v[i][0]:
                ans[i] = ans[i] + in_[i][j]
                num_v = num_v + 1
            else:
                break
        ans[i] = ans[i] / float(num_v)
    return ans

def smooth_l1_loss(logits, labels, sigma=1.0, dim=[1], use=True):
    sigma_2 = sigma ** 2
    diff = logits - labels
    abs_diff = tf.abs(diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    if use:
        loss = tf.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    else:
        loss = tf.pow(diff, 2) * (sigma_2 / 2.)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=dim))

    return loss

class compute(object):
    def __init__(self, label, label_raw, all_logits, is_training):
        """
        self.label = tf.reshape(label[:, 0 if(args.mission == 'theta') else 1], [args.batch_size])

        self.logits_cat = None
        self.prediction_cat = None
        self.loss = tf.Variable(0, dtype=tf.float32)
        for i in range(args.build_pyramid_layers):
            logits = all_logits['P%d' % (5-i)]
            if self.logits_cat == None:
                self.logits_cat = logits
                self.prediction_cat = tf.nn.softmax(logits)
            else:
                self.logits_cat = tf.concat([self.logits_cat, logits], axis=1)
                self.prediction_cat = tf.concat([self.prediction_cat, tf.nn.softmax(logits)], axis=1)
            self.loss = self.loss + tf.reduce_mean(loss_function(labels=smoothing_label(self.label, args.classes[args.mission]), logits=logits))

        self.prediction = tf.py_func(
            get_topk_mean,
            [tf.nn.top_k(self.prediction_cat, args.build_pyramid_layers*2)[0],
            tf.cast(tf.nn.top_k(self.prediction_cat, args.build_pyramid_layers*2)[1] % args.classes[args.mission], tf.float32)],
            tf.float32
        )

        self.classes_error_distribution = tf.cast(self.label, tf.float32) - self.prediction
        self.classes_error = tf.reduce_mean(tf.abs(self.classes_error_distribution))
        self.angular_error_distribution = self.classes_error_distribution / (float(args.classes_theta) / 0.565 / 90.0 if(args.mission == 'theta') else float(args.classes_phi) / 0.7 / 90.0)
        self.angular_error = tf.reduce_mean(tf.abs(self.angular_error_distribution))
        """
        self.label = label
        self.label_raw = label_raw
        self.loss = tf.Variable(0, dtype=tf.float32)
        self.angular_error_distribution = []
        for i in range(args.build_pyramid_layers):
            logits = all_logits['P%d' % (4-i)]
            self.loss = self.loss + smooth_l1_loss(logits=logits, labels=self.label_raw)
            # 标签对应的是弧度值站pi/2的百分比，换言之被限定在了-1~+1之间，返回原先的弧度值要*pi/2
            self.angular_error_distribution.append(self._predict_with_label(predict=logits, label=self.label_raw))

        # self.angular_error_distribution = tf.concat(self.angular_error_distribution, axis=0)
        self.angular_error_distribution = self.angular_error_distribution[0]
        self.angular_error = tf.reduce_mean(self.angular_error_distribution)

    def _predict_with_label(self, predict, label):

        def _2_to_3(radian):
            theta = radian[:, 0]; phi = radian[:, 1]
            x = (-1.)*tf.cos(theta)*tf.sin(phi)
            y = (-1.)*tf.sin(theta)
            z = (-1.)*tf.cos(theta)*tf.cos(phi)
            norm = tf.sqrt(x*x+y*y+z*z)
            return x, y, z, norm
        predict = predict*math.pi/2.0
        label = label*math.pi/2.0
        p_x, p_y, p_z, p_norm = _2_to_3(predict)
        l_x, l_y, l_z, l_norm = _2_to_3(label)
        angle = (p_x*l_x+p_y*l_y+p_z*l_z)/(p_norm*l_norm)
        return tf.acos(angle)*180/math.pi

def train():

    print(tf.__version__)
    print(('Do' if (args.build_pyramid) else 'Do not') + ' build pyramid!')

    with tf.Graph().as_default():

        Data = input_dataset.Dataset_reader(data_type=args.data_type,
                                            eyeball=args.eyeball,
                                            batch_size=args.batch_size,
                                            leave_one=args.leave_one,
                                            reduce_mean=args.reduce_mean,
                                            image_height=args.image_height,
                                            image_width=args.image_width,
                                            classes_theta=args.classes_theta,
                                            classes_phi=args.classes_phi)

        image = tf.placeholder(tf.float32, [args.batch_size, args.image_height, args.image_width, 3])
        label = tf.placeholder(tf.float32, [args.batch_size, 2])
        label_raw = tf.placeholder(tf.float32, [args.batch_size, 2])
        dropout_keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)

        logits, end_points = arch_net(image, is_training, dropout_keep_prob)
        
        variables_to_restore,variables_to_train = g_parameter(args.mission, args.net_name)

        g = compute(label, label_raw, logits, is_training)
        slim.losses.add_loss(g.loss)
        total_loss = slim.losses.get_total_loss()

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
        num_epoch = tf.get_variable('num_epoch', [], trainable=False, dtype=tf.float32)
        num_epoch = tf.cast(global_step, tf.float32) * args.batch_size / (Data.train_nums)
        lr = tf.train.piecewise_constant(num_epoch, boundaries=lr_boundaries, values=lr_values)

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', g.loss)
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('learning_rate', lr)

        
        with tf.name_scope('error'):
            tf.summary.scalar('angular_error', g.angular_error)
        
        var_list = variables_to_train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            # Adam
            # train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss, var_list=var_list, global_step = global_step, name='Adam')
            # SGD
            train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss, var_list=var_list, global_step=global_step, name='GD')
        saver_list = tf.global_variables()

        init = tf.global_variables_initializer()
        saver_restore = tf.train.Saver(saver_list)
        saver_train = tf.train.Saver(saver_list, max_to_keep = 100)
        merged = tf.summary.merge_all()

        def train_once(sess, mission, summary_writer_train):
            
            image_batch, label_batch, label_raw_batch = sess.run([Data.train_batch[0], Data.train_batch[1], Data.train_batch[2]])

            feed_dict_train = {image:image_batch, label:label_batch, label_raw:label_raw_batch, is_training:True, dropout_keep_prob:args.dropout_keep_prob}

            start_time = time.time()
            summary, loss_value, step, epoch, _ = sess.run([merged,
                                                            g.loss,
                                                            global_step,
                                                            num_epoch,
                                                            train_op],
                                                            feed_dict=feed_dict_train)
            end_time = time.time()
            sec_per_batch = end_time - start_time
            examples_per_sec = float(args.batch_size) / sec_per_batch

            summary_writer_train.add_summary(summary, int(1000 * epoch))

            print('epoch %d step %d of %s with loss %.4f(%.1f examples/sec; %.3f sec/batch)' % (int(epoch), step, mission, loss_value, examples_per_sec, sec_per_batch))

            return epoch, step

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer_train = tf.summary.FileWriter(logdir=summary_dir + '/train', graph=sess.graph)
            summary_writer_eval = tf.summary.FileWriter(logdir=summary_dir + '/eval')
            Best_answer = 180

            ckpt = tf.train.get_checkpoint_state(restore_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if args.restore_step == '0':
                    temp_dir = ckpt.model_checkpoint_path
                else:
                    temp_dir = ckpt.model_checkpoint_path.split('-')[0] + '-' + args.restore_step
                temp_step = int(temp_dir.split('-')[1])
                print('Restore the global parameters in the step %d!' % (temp_step)) 
                saver_restore.restore(sess, temp_dir)
            else:
                print('Initialize the global parameters')
                init.run()
 
            eval_epoch = 0
            test_epoch = 0
            checkpoint_epoch = 0

            while(args.trainable == '1'):
                epoch, step = train_once(sess, args.mission, summary_writer_train)

                if epoch > eval_epoch + 0.0025:
                    image_batch, label_batch, label_raw_batch = sess.run([Data.eval_batch[0], Data.eval_batch[1], Data.eval_batch[2]])
                    feed_dict_eval = {image:image_batch, label:label_batch, label_raw:label_raw_batch, is_training:False, dropout_keep_prob:1.0}
                    summary, loss_value = sess.run([merged, total_loss], feed_dict=feed_dict_eval)
                    summary_writer_eval.add_summary(summary, (1000 * epoch))
                    eval_epoch = epoch
                
                if epoch > test_epoch + 0.1:
                    angular_error = []
                    for i in range(int(Data.eval_nums / args.batch_size)):
                        image_batch, label_batch, label_raw_batch = sess.run([Data.eval_batch[0], Data.eval_batch[1], Data.eval_batch[2]])
                        feed_dict_eval = {image:image_batch, label:label_batch, label_raw:label_raw_batch, is_training:False, dropout_keep_prob:1.0}
                        angular_error_distribution = sess.run([g.angular_error_distribution], feed_dict=feed_dict_eval)
                        angular_error.append(angular_error_distribution)
                    angular_error = np.concatenate(angular_error, axis=0)
                    eval_dataset_angular_error = np.mean(angular_error)

                    if Best_answer > eval_dataset_angular_error:
                        Best_epoch = epoch
                        Best_step = step + 1
                        Best_answer = eval_dataset_angular_error

                    format_str = 'epoch %d step %d eval_dataset_angular_error = %.4f'
                    f = open(draw_dir + '/evaluation.txt', 'a')
                    print(format_str % (int(epoch), step, eval_dataset_angular_error), file=f)
                    print(angular_error, file=f)
                    f.close()
                    test_epoch = epoch
                
                if epoch > checkpoint_epoch + 2:
                    saver_train.save(sess, checkpoint_dir, global_step=step)
                    checkpoint_epoch = epoch
                
                if epoch >= args.epoch:
                    format_str = 'Best epoch %d step %d eval_dataset_angular_error = %.4f'
                    f = open(draw_dir + '/evaluation.txt', 'a')
                    print(format_str % (int(Best_epoch), Best_step, Best_answer), file=f)
                    f.close()
                    break
                
            summary_writer_train.close()
            summary_writer_eval.close()
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    print ("-----------------------------train.py start--------------------------")
    train()
