class compute(object):
    def __init__(self, label, label_raw, logits_theta, logits_phi, is_training, is_theta, is_phi):

        self.prediction_theta = tf.argmax(tf.nn.softmax(logits_theta), 1)
        self.prediction_phi = tf.argmax(tf.nn.softmax(logits_phi), 1)
        self.prediction_theta_raw = tf.reshape((tf.cast(self.prediction_theta, tf.float32) + 0.5) / float(args.classes_theta) * (0.565), [args.batch_size]) - 0.365
        self.prediction_phi_raw = tf.reshape((tf.cast(self.prediction_phi, tf.float32) + 0.5) / float(args.classes_phi) * (0.7), [args.batch_size]) - 0.3
        self.prediction_theta_raw = self.prediction_theta_raw * math.pi / 2.0
        self.prediction_phi_raw = self.prediction_phi_raw * math.pi / 2.0
        self.prediction_x = tf.cos(self.prediction_theta_raw) * tf.sin(self.prediction_phi_raw)
        self.prediction_y = tf.sin(self.prediction_theta_raw)
        self.prediction_z = tf.cos(self.prediction_theta_raw) * tf.cos(self.prediction_phi_raw)

        self.label_theta = tf.reshape(label[:, 0], [args.batch_size])
        self.label_phi = tf.reshape(label[:, 1], [args.batch_size])
        self.label_theta_raw = tf.reshape(label_raw[:, 0], [args.batch_size])
        self.label_phi_raw = tf.reshape(label_raw[:, 1], [args.batch_size])
        self.label_theta_raw = self.label_theta_raw * math.pi / 2.0
        self.label_phi_raw = self.label_phi_raw * math.pi / 2.0
        self.label_x = tf.cos(self.label_theta_raw) * tf.sin(self.label_phi_raw)
        self.label_y = tf.sin(self.label_theta_raw)
        self.label_z = tf.cos(self.label_theta_raw) * tf.cos(self.label_phi_raw)

        self.loss_angular_error_distribution = tf.acos(tf.clip_by_value(self.prediction_x * self.label_x +
                                                                        self.prediction_y * self.label_y +
                                                                        self.prediction_z * self.label_z, -1, 1)) / math.pi * 180.0
        self.loss_angular_error = tf.reduce_mean(self.loss_angular_error_distribution)
        self.loss_theta_softmax = tf.reduce_mean(loss_function(labels=smoothing_label(self.label_theta, args.classes_theta), logits=logits_theta))
        self.loss_phi_softmax = tf.reduce_mean(loss_function(labels=smoothing_label(self.label_phi, args.classes_phi), logits=logits_phi))

        self.loss = is_theta * self.loss_theta_softmax + is_phi * self.loss_phi_softmax
        
        self.label_aux = (self.label_theta // 10) * (args.classes_phi // 10) + (self.label_phi // 10)
        self.loss_aux_softmax = tf.reduce_mean(loss_function(labels=smoothing_label(self.label_aux, (args.classes_theta // 10) * (args.classes_phi // 10)), logits=logits_aux))
        self.loss = self.loss * (1 - args.aux_factor) + self.loss_aux_softmax * args.aux_factor

        self.classes_error_theta_distribution = tf.cast(self.label_theta - self.prediction_theta, tf.float32)
        self.classes_error_phi_distribution = tf.cast(self.label_phi - self.prediction_phi, tf.float32)

        self.classes_error_theta = tf.reduce_mean(tf.abs(self.classes_error_theta_distribution))
        self.classes_error_phi = tf.reduce_mean(tf.abs(self.classes_error_phi_distribution))

def draw_histgram(distribution_value, xlabel, ylabel, title, format_str, draw_dir, epoch, mean_value):
    plt.hist(distribution_value, bins=360, facecolor='blue', edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)                
    plt.savefig(format_str % (draw_dir, epoch, mean_value))
    plt.close('all')
    return True

def draw_layers(visual_layer_value, visual_layer_name, epoch):
    print('draw layers in epoch %d!' % epoch)
    visual_layer_dir = visual_dir + '/' + str(epoch) + '/'
    if not os.path.exists(visual_layer_dir):
        os.makedirs(visual_layer_dir)
    for i, key in enumerate(visual_layer_name):
        key = key.replace('/', '_')
        visual_layer = visual_layer_value[i]
        shape = visual_layer.shape
        channel = shape[0]
        print('draw %s with %d channels in size of %d X %d' % (key, shape[0], shape[1], shape[2])) 
        rows = 10
        cols = math.ceil(float(channel)/rows)
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols, rows))
        for r in range(rows):
            for c in range(cols):
                index = r * cols + c
                if index < channel:
                    vmin = visual_layer[index].min()
                    vmax = visual_layer[index].max()
                    axs[r, c].imshow(visual_layer[index], interpolation='bilinear', vmin=vmin, vmax=vmax)
                axs[r, c].set_axis_off()
        plt.savefig('%s%s.png' % (visual_layer_dir, key))
        plt.close('all')
    return True