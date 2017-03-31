from __future__ import division

import tensorflow as tf
import BasicConvLSTMCell

class ResNet(object):
	def __init__(self, input_conf=[[3,2,16,8],[4,2,16,8],[4,2,16,8],8], batch_size=32, layer={}, layer_param={}):
		# layer = ['conv', 'res_net', 'conv']
		# layer_param = [ [[3,3], [1,1,1,1], 64],
		# [ 3, [ [[3,3], [1,1,1,1], 64], [[3,3], [1,1,1,1], 64] ] ],
		# [[3,3], [1,1,1,1], 2] ]
		self.input_conf = input_conf
		self.nb_flow = self.input_conf[0][1]
		self.row = self.input_conf[0][2]
		self.col = self.input_conf[0][3]
		self.batch_size = batch_size
		self.layer = layer
		self.layer_param = layer_param
		self.x_c = tf.placeholder(tf.float32, [None, self.row, self.col, self.input_conf[0][0]*self.nb_flow])
		self.x_p = tf.placeholder(tf.float32, [None, self.row, self.col, self.input_conf[1][0]*self.nb_flow])
		self.x_t = tf.placeholder(tf.float32, [None, self.row, self.col, self.input_conf[2][0]*self.nb_flow])
		# for external input
		self.x_ext = tf.placeholder(tf.float32, [None, 8])
		#conf = self.input_conf[0]
		self.y = tf.placeholder(tf.float32, [None, self.row, self.col, self.nb_flow])

		self.weight_initializer = tf.contrib.layers.xavier_initializer()
		self.const_initializer = tf.constant_initializer()
		
	def conv(self, inputs, filter, strides, output_features, padding, idx):
		# param: filter, strides, output_features
		with tf.variable_scope('conv_{0}'.format(idx)) as scope:
			in_channels = inputs.get_shape().as_list()[3]
			w = tf.get_variable('w', [filter[0], filter[1], in_channels, output_features], initializer=self.weight_initializer)
			b = tf.get_variable('b', [output_features], initializer=self.const_initializer)
			y = tf.nn.conv2d(inputs, w, strides=strides, padding=padding)
			y_b = tf.nn.bias_add(y, b, name='wx_plus_b')
			y_relu = tf.nn.relu(y_b, name='out_conv_{0}'.format(idx))
			return y_relu

	def conv_transpose(self, inputs, filter, strides, output_features, padding, idx):
		with tf.variable_scope('conv_transpose_{0}'.format(idx)) as scope:
			in_channels = inputs.get_shape().as_list()[3]
			w = tf.get_variable('w', [filter[0], filter[1], output_features, in_channels], initializer=self.weight_initializer)
			b = tf.get_variable('b', [output_features], initializer=self.const_initializer)
			output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*strides[1], tf.shape(inputs)[2]*strides[2], output_features])
			y = tf.nn.conv2d_transpose(inputs, w, output_shape, strides=strides, padding=padding)
			y_b = tf.nn.bias_add(y, b, name='wx_plus_b')
			y_relu = tf.nn.relu(y_b, name='out_conv_transpose_{0}'.format(idx))
			return y_relu

	def res_unit(self, x, res_param, idx):
		res = x
		with tf.variable_scope('res_unit_{0}'.format(idx)) as scope:
			for i in range(len(res_param)):
				# TODO:
				# batch normalization
				res = self.conv(res, res_param[i][0], res_param[i][1], res_param[i][2], padding='SAME', idx=i)
		return x+res

	def res_net(self, x, unit_num, res_param, idx):
		y_ = x
		#unit_num = unit_num[0]
		with tf.variable_scope('res_net_{0}'.format(idx)) as scope:
			for i in range(unit_num):
				y_ = self.res_unit(y_, res_param=res_param, idx=i)
		return y_

	
	def fusion(self, x, idx):
		# x: [batch_size, row, col, nb_flow]
		with tf.variable_scope('fusion_input_{0}'.format(idx)) as scope:
			shape = x.get_shape().as_list()
			w = tf.get_variable('w', [shape[1], shape[2]], initializer=self.weight_initializer)
			w_extend = tf.expand_dims(w,axis=-1)
			return tf.multiply(w_extend, x)

	def build_model(self):
		x = [self.x_c, self.x_p, self.x_t, self.x_ext]
		y = self.y
		#input_conf = self.input_conf
		layer = self.layer
		param = self.layer_param
		y_all = []
		for i in range(len(x)-1):
			# i: inputs num
			with tf.variable_scope('input{0}'.format(i)):
				y_ = x[i]
				for l_i in range(len(layer)):
					# l_i: layers num of input i
					if layer[l_i]=='conv':
						y_ = self.conv(y_, param[l_i][0], param[l_i][1], param[l_i][2], padding='SAME', idx=l_i)
					if layer[l_i]=='res_net':
						y_ = self.res_net(y_, param[l_i][0], param[l_i][1], idx=l_i)
			# fusion
			# y_: [batch_size, row, col, channel]
			y_ = self.fusion(y_, idx=i)
			y_all.append(y_)
		# sum fusion
		y_all = tf.stack(y_all)
		print(y_all.get_shape().as_list())
		y_sum = tf.reduce_sum(y_all, axis=0, name='y_main')
		# external
		#print(i)
		y_ext = tf.layers.dense(x[-1], units=10, activation=tf.nn.relu, use_bias=True, 
			kernel_initializer=self.weight_initializer, bias_initializer=self.const_initializer,
			name='external_dense_1')
		y_ext = tf.layers.dense(y_ext, units=self.nb_flow*self.row*self.col, 
			activation=tf.nn.relu, use_bias=True, 
			kernel_initializer=self.weight_initializer, bias_initializer=self.const_initializer,
			name='external_dense_2')
		y_ext = tf.reshape(y_ext, [-1, self.row, self.col, self.nb_flow], name='y_ext')
		# y_sum + y_ext
		y_out = tf.nn.relu(tf.add(y_sum, y_ext), name='y_out')
		# compute loss
		loss = 2*tf.nn.l2_loss(y-y_out)
		return y_out, loss
		


