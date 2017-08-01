import tensorflow as tf
import shintb.utils.graph_components as gc



class GraphDrawer:
    def __init__(self, config):

        self.config = config

        print("Start Graph Drawing ... ")
        self.draw_placeholders()
        self.draw_base_network()
        self.draw_textbox_layer()
        self.draw_predictions()
        self.draw_loss_function()
        self.draw_optimizer()
        self.init_op = self.draw_init_op()
        self.draw_summary_op()
        print("Complete !!!")

    def draw_placeholders(self):
        c = self.config
        print(">>>Draw placeholders...", end=' ')

        with tf.name_scope(name = "placeholders") as scope:
            self.imgs_ph = tf.placeholder(dtype = tf.float32, shape = [None, 300, 300, 3], name =  "X")

            #true_conf_ph = tf.placeholder(dtype = tf.float32, shape = [None, total_boxes, 2], name = "true_conf")
            self.true_loc_ph = tf.placeholder(dtype = tf.float32, shape = [None, c["total_boxes"], 4], name = "true_loc")
            self.positive_ph = tf.placeholder(dtype = tf.float32, shape = [None, c["total_boxes"]], name = "positive")
            self.negative_ph = tf.placeholder(dtype = tf.float32, shape = [None ,c["total_boxes"]], name = "negative")

            self.learning_rate_ph = tf.placeholder(dtype = tf.float32, name = "learning_rate" )
            self.batch_norm_ph = tf.placeholder(tf.bool, name = "batch_norm_switch")

        print("Done!")

    def draw_base_network(self):
        print(">>>Draw base network...", end = " ")

        with tf.name_scope(name = "preprocessing") as scope:
            VGG_MEAN = {'R': 123.68, 'G': 116.779, 'B': 103.939}

            x = self.imgs_ph * 225.0
            blue, green, red = tf.split(x, 3, 3)
            x = tf.concat([blue - VGG_MEAN['B'], green - VGG_MEAN["G"], red - VGG_MEAN["R"]], 3)

            self.input = x

        with tf.name_scope(name="base_network") as scope:
            self.conv1_1 = gc.conv2d(self.input, 3, 64, name='conv1_1')  # 300
            self.conv1_2 = gc.conv2d(self.conv1_1, 64, 64, name='conv1_2')  # 300
            self.pool1 = gc.maxPool(self.conv1_2, name='pool1')  # 150
            self.conv2_1 = gc.conv2d(self.pool1, 64, 128, name='conv2_1')  # 150
            self.conv2_2 = gc.conv2d(self.conv2_1, 128, 128, name='conv2_2')  # 150
            self.pool2 = gc.maxPool(self.conv2_2, name='pool2')  # 75
            self.conv3_1 = gc.conv2d(self.pool2, 128, 256, name='conv3_1')  # 75
            self.conv3_2 = gc.conv2d(self.conv3_1, 256, 256, name='conv3_2')  # 75
            self.conv3_3 = gc.conv2d(self.conv3_2, 256, 256, name='conv3_3')  # 75
            self.pool3 = gc.maxPool(self.conv3_3, name='pool3')  # 38
            self.conv4_1 = gc.conv2d(self.pool3, 256, 512, name='conv4_1')  # 38
            self.conv4_2 = gc.conv2d(self.conv4_1, 512, 512, name='conv4_2')  # 38
            self.conv4_3 = gc.conv2d(self.conv4_2, 512, 512, name='conv4_3')  # 38
            self.pool4 = gc.maxPool(self.conv4_3, name='pool4')  # 19
            self.conv5_1 = gc.conv2d(self.pool4, 512, 512, name='conv5_1')  # 19
            self.conv5_2 = gc.conv2d(self.conv5_1, 512, 512, name='conv5_2')  # 19
            self.conv5_3 = gc.conv2d(self.conv5_2, 512, 512, name='conv5_3')  # 19
            self.pool5 = gc.maxPool(self.conv5_3, stride=1, kernel=3, name='pool5')  # 19
            self.conv6 = gc.conv2d(self.pool5, 512, 1024, name='conv6')  # 19
            self.conv7 = gc.conv2d(self.conv6, 1024, 1024, kernel=[1, 1], name='conv7')  # 19
            self.conv8_1 = gc.conv2d(self.conv7, 1024, 256, kernel=[1, 1], name='conv8_1')  # 19
            self.conv8_2 = gc.conv2d(self.conv8_1, 256, 512, strides=[2, 2], name='conv8_2')  # 10
            self.conv9_1 = gc.conv2d(self.conv8_2, 512, 128, kernel=[1, 1], name='conv9_1')  # 10
            self.conv9_2 = gc.conv2d(self.conv9_1, 128, 256, strides=[2, 2], name='conv9_2')  # 5
            self.conv10_1 = gc.conv2d(self.conv9_2, 256, 128, kernel=[1, 1], name='conv10_1')  # 5
            self.conv10_2 = gc.conv2d(self.conv10_1, 128, 256, strides=[2, 2], name='conv10_2')  # 3
            self.pool6 = tf.nn.avg_pool(self.conv10_2, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")  # 1

        print("Done!")

    def draw_textbox_layer(self):
        print(">>>Draw textbox layers(output layers)...", end=" ")
        with tf.name_scope(name="textbox_layer") as scope :
            self.out1 = gc.conv2d(self.conv4_3, 512, 72, kernel=[1, 5],bn = True, trainPhase=self.batch_norm_ph,  name='out1')
            self.out2 = gc.conv2d(self.conv7, 1024, 72, kernel=[1, 5], bn = True, trainPhase=self.batch_norm_ph,  name='out2')
            self.out3 = gc.conv2d(self.conv8_2, 512, 72, kernel=[1, 5], bn = True, trainPhase=self.batch_norm_ph, name='out3')
            self.out4 = gc.conv2d(self.conv9_2, 256, 72, kernel=[1, 5], bn = True, trainPhase=self.batch_norm_ph, name='out4')
            self.out5 = gc.conv2d(self.conv10_2, 256, 72, kernel=[1, 5], bn = True, trainPhase=self.batch_norm_ph, name='out5')
            self.out6 = gc.conv2d(self.pool6, 256, 72, kernel=[1, 1], bn = True, trainPhase=self.batch_norm_ph, name='out6')

        print("Done!")

    def draw_predictions(self):
        print(">>>Draw predicted confidences / predicted locations ...", end=" ")

        outs = [self.out1, self.out2, self.out3, self.out4, self.out5, self.out6]
        reshaped_outs = []
        for out in outs :
            w = out.get_shape().as_list()[2]
            h = out.get_shape().as_list()[1]
            reshaped = tf.reshape(out, [-1, w*h*12, 6])
            #out : [-1, h, w, 72]
            #reshaped : [-1, w*h*12, 6]
            reshaped_outs.append(reshaped)

        pred_boxes = tf.concat(reshaped_outs, 1)  # [?, 23280, 6]

        with tf.name_scope(name="predictions") as scope :
            self.pred_conf , self.pred_loc = tf.split(pred_boxes, [2,4], axis = 2)
            #self.pred_conf : [?, 23280 , 2]
            #self.pred_loc : [? ,23280, 4]

        print("Done!")

    def draw_loss_function(self):
        print(">>>Draw loss function ...", end=" ")
        # [?, 23280, 2]
        pred_conf_softmax =  tf.nn.softmax(self.pred_conf, -1)
        # [?, 23280, 1], [?, 23280, 1]
        text_conf , background_conf = tf.split(pred_conf_softmax, 2, 2)
        # [?, 23280]
        text_conf = tf.reshape(text_conf, [-1, text_conf.get_shape().as_list()[1]] )
        # [?, 23280]
        background_conf = tf.reshape(background_conf, [-1, background_conf.get_shape().as_list()[1]])
        self.conf_loss =  - tf.reduce_sum(tf.log(text_conf)*self.positive_ph) - tf.reduce_sum(tf.log(background_conf)*self.negative_ph)

        loc_loss = tf.reduce_sum(gc.smooth_l1(self.pred_loc - self.true_loc_ph), reduction_indices= 2) * self.positive_ph
        self.loc_loss = tf.reduce_sum(loc_loss, reduction_indices= 1) / (tf.reduce_sum(self.positive_ph, reduction_indices = 1) + 1e-5)
        # 1e-5 는 분모가 0일 경우를 방지

        self.total_loss = tf.reduce_mean(self.conf_loss + self.loc_loss)


        print("Done!")

    def draw_optimizer(self):
        print(">>>Draw optimizer and train ...", end=" ")

        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.train = self.optimizer.minimize(self.total_loss, self.global_step)

        print("Done!")
    def draw_init_op(self):
        init_op = tf.global_variables_initializer()
        return init_op

    def draw_summary_op(self):
        print(">>>Draw summary op ...", end=" ")
        tf.summary.scalar("loss/conf_loss", tf.reshape(tf.reduce_sum(self.conf_loss), shape=[]))
        tf.summary.scalar("loss/loc_loss", tf.reshape(tf.reduce_sum(self.loc_loss), shape=[]))
        tf.summary.scalar("loss/total_loss",  tf.reshape(self.total_loss, shape=[]))

        self.summaries = tf.summary.merge_all()
        print("Done!")

