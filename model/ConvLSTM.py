from __future__ import division

import tensorflow as tf
import BasicConvLSTMCell

class ConvLSTM(object):
    def __init__(self, input_dim=[64,64,2], batch_size=32, layer={}, layer_param={}, input_steps=10, output_steps=10, reg_lambda=0.02):
        #self.input_dim = input_dim
        self.input_row = input_dim[0]
        self.input_col = input_dim[1]
        self.input_channel = input_dim[2]

        self.batch_size = batch_size
        self.seq_length = input_steps + output_steps
        self.input_steps = input_steps
        self.output_steps = output_steps

        self.reg_lambda = reg_lambda

        self.encoder_layer = layer['encoder']
        self.decoder_layer = layer['decoder']
        self.encoder_layer_param = layer_param['encoder']
        self.decoder_layer_param = layer_param['decoder']

        # initialize conv_lstm cell
        self.encoder_conv_lstm = []
        self.encoder_state = []
        for i in range(len(self.encoder_layer)):
            if self.encoder_layer[i]=='conv_lstm':
                convLSTM = BasicConvLSTMCell.BasicConvLSTMCell(
                    self.encoder_layer_param[i][0], self.encoder_layer_param[i][1], self.encoder_layer_param[i][2],
                    state_is_tuple=True)
                self.encoder_conv_lstm.append(convLSTM)
                self.encoder_state.append(convLSTM.zero_state(self.batch_size))
        #self.init_state_encoder_conv_lstm = self.encoder_conv_lstm[0].zero_state(self.batch_size)

        self.decoder_conv_lstm = []
        self.decoder_state = []
        for i in range(len(self.decoder_layer)):
            if self.decoder_layer[i]=='conv_lstm':
                convLSTM = BasicConvLSTMCell.BasicConvLSTMCell(
                    self.decoder_layer_param[i][0], self.decoder_layer_param[i][1], self.decoder_layer_param[i][2],
                    state_is_tuple=True)
                self.decoder_conv_lstm.append(convLSTM)
                self.decoder_state.append(convLSTM.zero_state(self.batch_size))
        #self.init_state_decoder_conv_lstm = self.decoder_conv_lstm[0].zero_state(self.batch_size)

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        self.x = tf.placeholder(tf.float32, [None, self.input_steps, self.input_row, self.input_col, self.input_channel])
        self.y = tf.placeholder(tf.float32, [None, self.output_steps, self.input_row, self.input_col, self.input_channel])

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

    #def conv_lstm():
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

    def encoder(self, x, last_state, reuse=True):
        layer = self.encoder_layer
        param = self.encoder_layer_param
        y = x
        #state = self.encoder_state
        state = last_state
        with tf.variable_scope('encoder', reuse=reuse):
            # layer: ['conv', 'conv_lstm']
            conv_lstm_index = 0;
            for i in range(len(layer)):
                if layer[i]=='conv':
                    y = self.conv(y, param[i][0], param[i][1], param[i][2], padding='SAME', idx=i)
                if layer[i]=='conv_lstm':
                    y, state[conv_lstm_index] = self.encoder_conv_lstm[conv_lstm_index](y, state[conv_lstm_index], scope='conv_lstm_{0}'.format(i))
                    conv_lstm_index += 1
        return y, state

    def decoder(self, init_state, reuse=True):
        layer = self.decoder_layer
        param = self.decoder_layer_param
        y = None
        state = init_state
        with tf.variable_scope('decoder', reuse=reuse):
            conv_lstm_index = 0;
            for i in range(len(layer)):
                if layer[i]=='conv':
                    y = self.conv_transpose(y, param[i][0], param[i][1], param[i][2], padding='SAME', idx=i)
                if layer[i]=='conv_lstm':
                    y, state[conv_lstm_index] = self.decoder_conv_lstm[conv_lstm_index](y, state[conv_lstm_index], scope='conv_lstm_{0}'.format(i))
                    conv_lstm_index += 1
        return y, state


    def build_model(self):
        x = self.x
        y = self.y
        # x: [batch_size, seq_length, row, col, channel]
        #batch_size = tf.shape(x)[0]
        #self.init_state_encoder_conv_lstm = self.encoder_conv_lstm[0].zero_state(batch_size)
        # encoder
        state = self.encoder_state
        for t in range(self.input_steps):
            _, state = self.encoder(x[:, t, :, :, :], state, reuse=(t!=0))
        state_2 = state
        y_ = []
        for t in range(self.output_steps):
            out, state_2 = self.decoder(state_2, reuse=(t!=0))
            y_.append(out)
        y_ = tf.stack(y_)
        y_ = tf.transpose(y_, [1,0,2,3,4])
        loss = 2*tf.nn.l2_loss(y-y_[:,:,:,:,:])
        #return loss/tf.to_float(batch_size)
        # tf.sqrt(loss/tf.to_float(batch_size*seq_length*row*col*channel))
        # weighted loss
        step_weight = tf.get_variable('step_weight', [self.output_steps], initializer=self.const_initializer)
        step_weight = tf.nn.softmax(step_weight)
        square_loss = tf.reduce_mean(tf.square(y-y_), [0, 2, 3, 4])
        weighted_loss = tf.reduce_sum(tf.multiply(square_loss, step_weight)) + \
                        self.reg_lambda * tf.nn.l2_loss(step_weight)
        return y_, loss, weighted_loss, step_weight

    # def build_sampler(self):
    # 	x = self.x
    # 	y = self.y
    # 	#batch_size = tf.shape(x)[0]
    # 	state = self.encoder_state
    # 	for t in range(self.input_steps):
    # 		_, state = self.encoder(x[:, t, :, :, :], state)
    # 	state_2 = state
    # 	y_ = []
    # 	for t in range(self.output_steps):
    # 		out, state_2 = self.decoder(state_2)
    # 		y_.append(out)
    # 	y_ = tf.stack(y_)
    # 	y_ = tf.transpose(y_, [1,0,2,3,4])
    # 	loss = 2*tf.nn.l2_loss(y-y_[:,:,:,:,:])
    # 	return y_, loss


