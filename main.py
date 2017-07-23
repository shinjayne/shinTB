import tensorflow as tf
import numpy as np
import cv2



# 1. data prepare / default_box prepare / pos and neg prepare
# 2. draw graph
# 3. run


# constant
img_wh = 300

total_boxes = 23280

# hyperparams






# runtime

with tf.Session(graph = g) as sess :
    sess.run(tf.global_variables_initializer())

    while True :
        lr = learning_rate_maker(sess.run(global_step))



