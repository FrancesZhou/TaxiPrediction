#from __future__ import division
import numpy as np
import time
import os
import tensorflow as tf

class ModelSolver(object):
	def __init__(self, model, data, val_data, preprocessing, **kwargs):
		self.model = model
		self.data = data
		self.val_data = val_data
		self.preprocessing = preprocessing
		self.n_epochs = kwargs.pop('n_epochs', 10)
		self.batch_size = kwargs.pop('batch_size', 32)
		self.learning_rate = kwargs.pop('learning_rate', 0.000001)
		self.update_rule = kwargs.pop('update_rule', 'adam')
		self.model_path = kwargs.pop('model_path', './model/')
		self.save_every = kwargs.pop('save_every', 1)
		self.log_path = kwargs.pop('log_path', './log/')
		self.pretrained_model = kwargs.pop('pretrained_model', None)
		self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

		if self.update_rule == 'adam':
			self.optimizer = tf.train.AdamOptimizer
		elif self.update_rule == 'momentum':
			self.optimizer = tf.train.MomentumOptimizer
		elif self.update_rule == 'rmsprop':
			self.optimizer = tf.train.RMSPropOptimizer

		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
		if not os.path.exists(self.log_path):
			os.makedirs(self.log_path)

	def train(self):
		x = np.asarray(self.data['x'])
		#print('shape of x: '+x.shape())
		y = np.asarray(self.data['y'])
		x_val = np.asarray(self.val_data['x'])
		y_val = np.asarray(self.val_data['y'])

		# build graphs
		y_, loss = self.model.build_model()

		#tf.get_variable_scope().reuse_variables()
		#y_ = self.model.build_sampler()

		# train op
		with tf.name_scope('optimizer'):
			optimizer = self.optimizer(learning_rate=self.learning_rate)
			grads = tf.gradients(loss, tf.trainable_variables())
			grads_and_vars = list(zip(grads, tf.trainable_variables()))
			train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

		tf.get_variable_scope().reuse_variables()
		#y_ = self.model.build_sampler()
		# summary op
		tf.summary.scalar('batch_loss', loss)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
		for grad, var in grads_and_vars:
			tf.summary.histogram(var.op.name+'/gradient', grad)

		summary_op = tf.summary.merge_all()

		with tf.Session() as sess:
			tf.global_variables_initializer().run()
			summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
			saver = tf.train.Saver(tf.global_variables())
			if self.pretrained_model is not None:
				print "Start training with pretrained model..."
				saver.restore(sess, self.pretrained_model)

			#curr_loss = 0
			start_t = time.time()
			for e in range(self.n_epochs):
				curr_loss = 0
				for i in range(len(x)):
					feed_dict = {self.model.x: x[i,:,:,:,:], self.model.y: y[i,:,:,:,:]}
					_, l = sess.run([train_op, loss], feed_dict)
					curr_loss += l

					# write summary for tensorboard visualization
					if i%100 == 0:
						print("at epoch "+str(e)+', '+str(i))
						summary = sess.run(summary_op, feed_dict)
						summary_writer.add_summary(summary, e*x.shape[0] + i)
				t_rmse = np.sqrt(curr_loss/(np.prod(y.shape)))
				print("at epoch " + str(e) + ", train loss is " + str(curr_loss)+','+str(t_rmse)+','+ str(self.preprocessing.real_loss(t_rmse)))
				# validate
				val_loss = 0
				y_pred_all = np.ndarray(y_val.shape)
				for i in range(len(y_val)):
					feed_dict = {self.model.x: x_val[i,:,:,:,:], self.model.y: y_val[i,:,:,:,:]}
					y_p, l = sess.run([y_, loss], feed_dict=feed_dict)
					y_pred_all[i] = y_p
					val_loss += l

				# y_val : [batches, batch_size, seq_length, row, col, channel]
				rmse = np.sqrt(val_loss/(np.prod(y_val.shape)))
				print("at epoch " + str(e) + ", validate loss is " + str(self.preprocessing.real_loss(rmse)))
				print "elapsed time: ", time.time() - start_t

				if (e+1)%self.save_every == 0:
					save_name = self.model_path+'model'
					saver.save(sess, save_name, global_step=e+1)
					print "model-%s saved." % (e+1)

	def test(self, data, save_outputs=True):
		x = np.asarray(data['x'])
		y = np.asarray(data['y'])

		# build graphs
		y_, loss = self.model.build_model()

		#y = np.asarray(y)
		y_pred_all = np.ndarray(y.shape)
		
		#y_real = tf.convert_to_tensor(y)
		#loss = 2*tf.nn.l2_loss(y_real-y_)
		#summary_op = tf.summary.merge_all()

		with tf.Session() as sess:
			#tf.initialize_all_variables().run()
			#summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
			saver = tf.train.Saver()
			saver.restore(sess, self.test_model)
			start_t = time.time()
			#y_pred_all = np.ndarray(y.shape)
			t_loss = 0
			for i in range(len(y)):
				feed_dict = {self.model.x: x[i,:,:,:,:], self.model.y: y[i,:,:,:,:]}
				y_p, l = sess.run([y_, loss], feed_dict=feed_dict)
				y_pred_all[i] = y_p
				t_loss += l
				
			# y : [batches, batch_size, seq_length, row, col, channel]
			rmse = np.sqrt(t_loss/(np.prod(y.shape)))
			print("test loss is " + str(self.preprocessing.real_loss(rmse)))
			print("elapsed time: ", time.time() - start_t)
			if save_outputs:
				np.save('test_outputs.npy',y_pred_all)












