from __future__ import division

import numpy as np
import tensorflow as tf
import BasicConvLSTMCell

class AttResNet(object):
    def __init__(self, input_conf=[[3,2,16,8],[4,2,16,8],[4,2,16,8],[8]], att_inputs=[], att_nodes=1024, att_layer={}, att_layer_param={}, batch_size=32, layer={}, layer_param={}):
        # layer = ['conv', 'res_net', 'conv']
        # layer_param = [ [[3,3], [1,1,1,1], 64],
        # [ 3, [ [[3,3], [1,1,1,1], 64], [[3,3], [1,1,1,1], 64] ] ],
        # [[3,3], [1,1,1,1], 2] ]
        self.input_conf = input_conf
        self.nb_flow = self.input_conf[0][1]
        self.row = self.input_conf[0][2]
        self.col = self.input_conf[0][3]

        self.att_inputs = att_inputs
        self.att_nodes = att_nodes
        self.att_layer = att_layer
        self.att_layer_param = att_layer_param

        self.batch_size = batch_size
        self.layer = layer
        self.layer_param = layer_param
        self.x_c = tf.placeholder(tf.float32, [None, self.row, self.col, self.input_conf[0][0]*self.nb_flow])
        self.x_p = tf.placeholder(tf.float32, [None, self.row, self.col, self.input_conf[1][0]*self.nb_flow])
        self.x_t = tf.placeholder(tf.float32, [None, self.row, self.col, self.input_conf[2][0]*self.nb_flow])
        # for external input
        self.x_ext = tf.placeholder(tf.float32, [None, self.input_conf[-1][0]])
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

    def attention_layer(self, state):
        layer = self.att_layer
        param = self.att_layer_param
        # att_inputs: [cluster_num, row, col, channel]
        y = tf.convert_to_tensor(self.att_inputs, dtype=tf.float32)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        #print 'state shape:'
        #print state.get_shape().as_list()
        #h = tf.reshape(state, [state.get_shape().as_list()[0], -1])
        with tf.variable_scope('attention'):
            for i in range(len(layer)):
                if layer[i]=='conv':
                    y = self.conv(y, param[i][0], param[i][1], param[i][2], padding='SAME', idx=i)
            # attention and hidden state
            # y: [cluster_num, 16, 16, 16]
            # att: [cluster_num, att_num]
            y_shape = y.get_shape().as_list()
            att = tf.reshape(y, [y_shape[0], -1])
            # att_shape: [cluster_num, att_num]
            att_shape = att.get_shape().as_list()
            # h: [batch_size, row, col, channel] -> [batch_size, h_num]
            #h = tf.reshape(state, [state.get_shape().as_list()[0], -1])
            h = tf.reshape(state, [-1, np.prod(state.get_shape().as_list()[1:])])
            # h_shape: [batch_size, h_num]
            h_shape = h.get_shape().as_list()

            h_num = h_shape[1]
            att_num = att_shape[1]
            # att: [cluster_num, att_num]
            # h: [batch_size, h_num]
            with tf.variable_scope('att_hidden'):
                with tf.variable_scope('hidden'):
                    w = tf.get_variable('w', [h_num, self.att_nodes], initializer=self.weight_initializer)
                    # h_att : [batch_size, att_nodes]
                    h_att = tf.matmul(h, w)
                with tf.variable_scope('att'):
                    w = tf.get_variable('w', [att_num, self.att_nodes], initializer=self.weight_initializer)
                    # att_proj : [cluster_num, att_nodes]
                    att_proj = tf.matmul(att, w)
                b = tf.get_variable('b', [self.att_nodes], initializer=self.const_initializer)
                # [batch_size, cluster_num, att_nodes]
                # att_proj: [cluster_num, att_nodes]
                # tf.tile(tf.expand_dims(h_att, 1), [1, att_shape[0], 1]) -> [batch_size, cluster_num, att_nodes]
                att_h_plus = tf.nn.relu(att_proj + tf.tile(tf.expand_dims(h_att, 1), [1, att_shape[0], 1]) + b)
                #att_h_plus = tf.nn.relu(att_proj + tf.expand_dims(h_att, 1) + b)
                # att_h_plus: [batch_size, cluster_num, att_nodes]
                w_att = tf.get_variable('w_att', [self.att_nodes, 1], initializer=self.weight_initializer)
                out_att = tf.reshape(tf.matmul(tf.reshape(att_h_plus, [-1, self.att_nodes]), w_att), [-1, att_shape[0]])
                # out_att: [batch_size, cluster_num]
                alpha = tf.nn.softmax(out_att)
                # att: [cluster_num, att_num]
                # context: [batch_size, att_num]
                context = tf.reduce_sum(att * tf.tile(tf.expand_dims(alpha, -1), [1, 1, att_num]), 1, name='context')
                att_context = tf.reshape(context, [-1, y_shape[1], y_shape[2], y_shape[3]])
                #context = tf.reduce_sum(att * tf.expand_dims(alpha, 2), 1, name='context')
                #out_shape = y.get_shape().as_list()
                #out_shape[0] = h_shape[0]
                #att_context = tf.reshape(context, out_shape)
                return att_context, alpha

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
            # add attention
            if i == 0:
                y_ctx, _ = self.attention_layer(y_)
                y_ctx = self.fusion(y_ctx, idx=0.1)
                y_all.append(y_ctx)
        # sum fusion
        y_all = tf.stack(y_all)
        #print(y_all.get_shape().as_list())
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



