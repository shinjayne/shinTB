import tensorflow as tf


def weightVariable(shape):
	initial = tf.truncated_normal(shape ,stddev=0.1, name='weights')
	return tf.Variable(initial)

	#return tf.get_variable(name="W", shape=shape, initializer=tf.random_normal_initializer(mean=0.0, stddev = 0.1))

def biasVariable(shape):
	initial = tf.constant(0.1, shape=shape, name='biases')
	return tf.Variable(initial)
	#return tf.get_variable(name="B", shape=shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

def maxPool(input, stride=2, kernel=2, padding='SAME', name='pool'):
	return tf.nn.max_pool(input, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

def conv2d(input, inputNum, outputNum, kernel=[3, 3], strides=[1, 1], padding='SAME', bn=False, trainPhase=True, name='conv2d'):
	with tf.name_scope(name) as scope:
		W = weightVariable([kernel[0], kernel[1], inputNum, outputNum])
		b = biasVariable([outputNum])
		conv_out = tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding=padding)
		biased_out = tf.nn.bias_add(conv_out, b)
		out = tf.nn.relu(biased_out)
		if bn:
			out = tf.contrib.layers.batch_norm(out, center=False, is_training=trainPhase)
		return out

def smooth_l1(x):
	l2 = 0.5 * (x**2.0)
	l1 = tf.abs(x) - 0.5

	condition = tf.less(tf.abs(x), 1.0)
	re = tf.where(condition, l2, l1)

	return re