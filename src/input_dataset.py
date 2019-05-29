from PIL import Image
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

class Dataset_reader(object):
    def __init__(self, data_type, eyeball, batch_size, leave_one, reduce_mean,
                 image_height, image_width, classes_theta, classes_phi,
                 min_queue_examples=512):
        self.data_type = data_type
        self.eyeball = eyeball
        self.batch_size = batch_size
        self.leave_one = leave_one
        self.reduce_mean = reduce_mean
        self.image_height = 144
        self.image_width = 240
        self.image_size = [144, 240] 
        self.classes_theta = classes_theta
        self.classes_phi = classes_phi
        self.min_queue_examples = min_queue_examples
        self.data_file = '../dataset/'+data_type+'/'+str(self.leave_one)+'/'
        self.num_ones = {
            'left' :  [29961, 24143, 28019, 35075, 16831, 16577, 18448, 15509, 10701, 7995, 2810, 2982, 1609, 1498, 1500],
            'right' : [29961, 24143, 28019, 35075, 16831, 16577, 18448, 15509, 10701, 7995, 2810, 2982, 1609, 1498, 1500],
        }
        self.reader_mean()
        self.dataset_build()

    def reader_mean(self):
        load_data = sio.loadmat('../dataset/'+self.data_type+'/'+self.eyeball+'_mean.mat')
        self.mean_img = None
        self.train_nums = 0
        for i in range(0, 15, 1):
            if i == self.leave_one:
                self.eval_nums = self.num_ones[self.eyeball][i]
                continue;
            if self.mean_img is None:
                self.mean_img = load_data['mean_img_raw_%d' % i] * self.num_ones[self.eyeball][i]
            else:
                self.mean_img = self.mean_img + load_data['mean_img_raw_%d' % i] * self.num_ones[self.eyeball][i]
            self.train_nums = self.train_nums + self.num_ones[self.eyeball][i]
        self.mean_img = self.mean_img / float(self.train_nums)

    def read_image(self, filename_queue, name):

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'img_raw' : tf.FixedLenFeature([], tf.string),
            'label_raw' : tf.FixedLenFeature([], tf.string),
            }
        )

        eye = tf.decode_raw(features['img_raw'], tf.uint8)
        eye = tf.reshape(eye, [self.image_height, self.image_width, 3])
        eye = tf.cast(eye, tf.float32)
    
        label = tf.decode_raw(features['label_raw'], tf.float64)
        label = tf.reshape(label, [2])
        label = tf.cast(label, tf.float32)
    
        label_raw = label
        label = (label+1.)/2.
        
        return self.augmentation(eye, label, label_raw, name)

    def augmentation(self, eye, label, label_raw, name):
       
        def N_sample(min_v, max_v):
            p = np.random.randn()
            p = min(p, 3); p = max(p, -3)
            p = (p+3.0)/6.0
            return min_v+(max_v-min_v)*p
        if name == 'eval':
            eye = eye / 255.0
            return eye, label, label_raw
        """
        eye = tf.image.random_brightness(eye, max_delta=N_sample(0,0.25))
        bbox = c = tf.concat([tf.zeros([1, 1, 2]), tf.ones([1, 1, 2])], axis=2)
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(eye),
            bounding_boxes=bbox,
            aspect_ratio_range=[self.image_width/self.image_height,self.image_width/self.image_height],
            area_range=[0.975,1])
        eye = tf.slice(eye, bbox_begin, bbox_size)
        eye = tf.image.resize_images(eye, self.image_size, method=np.random.randint(4))
        eye = tf.reshape(eye, [self.image_height, self.image_width, 3])
        """
        eye = eye / 255.0
        return eye, label, label_raw

    def _tfrecords_to_batch(self, tfrecords_list, name):
        filename_queue = [tf.train.string_input_producer(i) for i in tfrecords_list]
        example_list = [self.read_image(i, name) for i in filename_queue]
        eye_batch, label_batch, label_raw_batch = tf.train.shuffle_batch_join(example_list,
                                                                              batch_size=self.batch_size,
                                                                              capacity=self.min_queue_examples+3*self.batch_size,
                                                                              min_after_dequeue=self.min_queue_examples,
                                                                              name=name)
        return (eye_batch, label_batch, label_raw_batch)

    def dataset_build(self):
        train_list = []
        eval_list = []
        for i in range(0, 15, 1):
            if i == self.leave_one:
                eval_list.append(['../dataset/LOPO_MPIIGaze/'+str(i)+'/'+self.eyeball+'_eval.tfrecords'])
            else:
                train_list.append(['../dataset/LOPO_MPIIGaze/'+str(i)+'/'+self.eyeball+'_train.tfrecords'])
        
        self.train_batch = self._tfrecords_to_batch(train_list, 'train')
        self.eval_batch = self._tfrecords_to_batch(eval_list, 'eval')

def test_input_dataset():
    D = Dataset_reader(data_type='LOPO_MPIIGaze',
                       eyeball='left',
                       batch_size=16,
                       leave_one=14,
                       image_height=144,
                       image_width=240,
                       classes_theta=90,
                       classes_phi=90,
                       reduce_mean=False)
    plt.imshow(D.mean_img.astype('uint8'))
    plt.savefig('../test/mean.jpg')
    plt.close()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        img, l, l_raw = sess.run([D.train_batch['2'][0], D.train_batch['2'][1], D.train_batch['2'][2]])
        for i in range(16):
            plt.imshow((np.squeeze(img[i])*255.0).astype('uint8'))
            plt.savefig('../test/%d.jpg' % i)
            plt.close()
            print(l[i][0], l[i][1])
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    test_input_dataset()
