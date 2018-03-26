from __future__ import division

import os
import numpy as np
import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, input_dim=[64, 64, 2], z_dim=[16, 16, 16], layer={}, layer_param={}):
        self.input_row = input_dim[0]
        self.input_col = input_dim[1]
        self.input_channel = input_dim[2]

        self.z_row = z_dim[0]
        self.z_col = z_dim[1]
        self.z_channel = z_dim[2]

        self.encoder_layer = layer['encoder']
        self.decoder_layer = layer['decoder']
        self.encoder_layer_param = layer_param['encoder']
        self.decoder_layer_param = layer_param['decoder']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        self.x = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[1, self.input_row, self.input_col, self.input_channel]), [None, self.input_row, self.input_col, self.input_channel])
        self.z = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[1, self.z_row, self.z_col, self.z_channel]), [None, self.z_row, self.z_col, self.z_channel])
        #self.y = tf.placeholder(tf.float32, [None, self.input_row, self.input_col, self.input_channel])

    def conv(self, inputs, filter, strides, output_features, padding, idx):
        # param: filter, strides, output_features
        with tf.variable_scope('conv_{0}'.format(idx)):
            in_channels = inputs.get_shape().as_list()[3]
            w = tf.get_variable('w', [filter[0], filter[1], in_channels, output_features], initializer=self.weight_initializer)
            b = tf.get_variable('b', [output_features], initializer=self.const_initializer)
            y = tf.nn.conv2d(inputs, w, strides=strides, padding=padding)
            y_b = tf.nn.bias_add(y, b, name='wx_plus_b')
            y_relu = tf.nn.relu(y_b, name='out_conv_{0}'.format(idx))
            return y_relu, w

    def conv_transpose(self, inputs, filter, strides, output_features, padding, idx, given_w=None):
        with tf.variable_scope('conv_transpose_{0}'.format(idx)):
            in_channels = inputs.get_shape().as_list()[3]
            if given_w is not None:
                w = given_w
            else:
                w = tf.get_variable('w', [filter[0], filter[1], output_features, in_channels], initializer=self.weight_initializer)
            b = tf.get_variable('b', [output_features], initializer=self.const_initializer)
            output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*strides[1], tf.shape(inputs)[2]*strides[2], output_features])
            y = tf.nn.conv2d_transpose(inputs, w, output_shape, strides=strides, padding=padding)
            y_b = tf.nn.bias_add(y, b, name='wx_plus_b')
            y_relu = tf.nn.relu(y_b, name='out_conv_transpose_{0}'.format(idx))
            return y_relu

    def encoder(self, x):
        layer = self.encoder_layer
        param = self.encoder_layer_param
        encoder_w = []
        y = x
        with tf.variable_scope('encoder'):
            # layer: ['conv', 'conv_lstm']
            for i in range(len(layer)):
                if layer[i]=='conv':
                    y, w = self.conv(y, param[i][0], param[i][1], param[i][2], padding='SAME', idx=i)
                    encoder_w.append(w)
                else:
                    continue
        return y, encoder_w

    def decoder(self, z):
        layer = self.decoder_layer
        param = self.decoder_layer_param
        y = z
        encoder_w = self.encoder_w
        with tf.variable_scope('decoder'):
            for i in range(len(layer)):
                if layer[i]=='conv':
                    y = self.conv_transpose(y, param[i][0], param[i][1], param[i][2], padding='SAME', idx=i, given_w=encoder_w[-i])
        return y

    def build_model(self):
        # x = self.x
        # x: [batch_size, row, col, channel]
        # encoder
        z, encoder_w = self.encoder(self.x)
        self.encoder_w = encoder_w
        # decoder
        x_ = self.decoder(z)
        #y_ = self.decoder(self.z)
        loss = 2*tf.nn.l2_loss(self.x-x_[:, :, :, :, :])
        return z, loss

    def train(self, data, batch_size, learning_rate, n_epochs, model_save_path):
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        self.model_path = model_save_path
        self.batch_size = batch_size
        # build model
        z, loss = self.build_model()
        # train op
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
        tf.get_variable_scope().reuse_variables()
        gpu_options = tf.GPUOptions(allow_growth=True)
        # train
        num_batch = int(np.ceil(data.shape[0] / float(batch_size)))
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            for epoch_i in range(n_epochs):
                curr_loss = 0
                for batch_i in np.random.permutation(num_batch):
                    x = data[batch_i*batch_size: min(data.shape[0], (batch_i + 1)*batch_size)]
                    _, l = sess.run([train_op, loss], feed_dict={self.x: x})
                    curr_loss += l
                mean_loss = curr_loss/data.shape[0]
                print 'at epoch %d, train loss is %f' % (epoch_i, mean_loss)
                # save model
                save_name = model_save_path + 'ae'
                saver.save(sess, save_name, global_step=epoch_i+1)

    def get_z(self, data, pretrained_model=None):
        z, loss = self.build_model()
        data_z = []
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver = tf.train.Saver(tf.global_variables())
            if pretrained_model is not None:
                print "load pretrained model..."
                pretrained_model_path = self.model_path + pretrained_model
                saver.restore(sess, pretrained_model_path)
            else:
                saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            recon_loss = 0
            for batch_i in range(int(np.ceil(data.shape[0]/float(self.batch_size)))):
                x = data[batch_i*self.batch_size: min(data.shape[0], (batch_i + 1)*self.batch_size)]
                z_batch, l = sess.run([z, loss], feed_dict={self.x: x})
                data_z.append(np.reshape(z_batch, [len(z_batch), -1]))
                recon_loss += l
            mean_loss = recon_loss/data.shape[0]
            print 'reconstruction loss is %f' % mean_loss
            data_z = np.concatenate(data_z, axis=0)
        return data_z

    def get_y(self, z_data, pretrained_model=None):
        y_ = self.decoder(self.z)
        y_data = []
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver = tf.train.Saver(tf.global_variables())
            if pretrained_model is not None:
                print "load pretrained model..."
                pretrained_model_path = self.model_path + pretrained_model
                saver.restore(sess, pretrained_model_path)
            else:
                saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            for i in range(len(z_data)):
                y_z = sess.run(y_, feed_dict={self.z: z_data[i]})
                y_data.append(y_z)
        return np.array(y_data)





