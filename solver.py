#from __future__ import division
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
sys.path.append('./util/')
from utils import *

class ModelSolver(object):
	def __init__(self, model, data, val_data, preprocessing, **kwargs):
		self.model = model
		self.data = data
		self.val_data = val_data
		self.preprocessing = preprocessing
		self.cross_val = kwargs.pop('cross_val', False)
		self.cpt_ext = kwargs.pop('cpt_ext', False)
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

	def train(self, test_data, test_1_to_n_data=[]):
		raw_x = x = self.data['x']
		raw_y = y = self.data['y']
		x_val = self.val_data['x']
		y_val = self.val_data['y']
		# x = np.asarray(self.data['x'])
		# y = np.asarray(self.data['y'])
		# x_val = np.asarray(self.val_data['x'])
		# y_val = np.asarray(self.val_data['y'])
		#print('shape of x: '+x.shape())
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
				# cross validation
				if self.cross_val:
					x, x_val, y, y_val = train_test_split(raw_x, raw_y, test_size=0.1, random_state=50)
					#print(np.array(x).shape)
					#print(np.array(y).shape)
				for i in range(len(x)):
					if self.cpt_ext:
						#print(x[i][0].shape)
						#print(x[i][1].shape)
						#print(x[i][2].shape)
						#print(x[i][3].shape)
						#print(np.array(y[i]).shape)
						feed_dict = {self.model.x_c: np.array(x[i][0]), self.model.x_p: np.array(x[i][1]), self.model.x_t: np.array(x[i][2]), 
									self.model.x_ext: np.array(x[i][3]), 
									self.model.y: np.array(y[i])}
					else:
						feed_dict = {self.model.x: np.array(x[i]), self.model.y: np.array(y[i])}
					_, l = sess.run([train_op, loss], feed_dict)
					curr_loss += l

					# write summary for tensorboard visualization
					if i%100 == 0:
						print("at epoch "+str(e)+', '+str(i))
						summary = sess.run(summary_op, feed_dict)
						summary_writer.add_summary(summary, e*len(x) + i)
				print(np.array(y).shape)
				#compute counts of all regions
				t_count = 0
				for c in range(len(y)):
					#print(np.array(y[c]).shape)
					t_count += np.prod(np.array(y[c]).shape)
				t_rmse = np.sqrt(curr_loss/t_count)
				#t_rmse = np.sqrt(curr_loss/(np.prod(np.array(y).shape)))
				print("at epoch " + str(e) + ", train loss is " + str(curr_loss) + ' , ' + str(t_rmse) + ' , ' + str(self.preprocessing.real_loss(t_rmse)))
				# validate
				val_loss = 0
				for i in range(len(y_val)):
					if self.cpt_ext:
						feed_dict = {self.model.x_c: np.array(x_val[i][0]), self.model.x_p: np.array(x_val[i][1]), self.model.x_t: np.array(x_val[i][2]), 
									self.model.x_ext: np.array(x_val[i][3]), 
									self.model.y: np.array(y_val[i])}
					else:
						feed_dict = {self.model.x: x_val[i], self.model.y: y_val[i]}
					_, l = sess.run([y_, loss], feed_dict=feed_dict)
					val_loss += l

				# y_val : [batches, batch_size, seq_length, row, col, channel]
				print(np.array(y_val).shape)
				v_count = 0
				for v in range(len(y_val)):
					#print(np.array(y_val[v]).shape)
					v_count += np.prod(np.array(y_val[v]).shape)
				rmse = np.sqrt(val_loss/v_count)
				#rmse = np.sqrt(val_loss/(np.prod(np.array(y_val).shape)))
				print("at epoch " + str(e) + ", validate loss is " + str(val_loss) + ' , ' + str(rmse) + ' , ' + str(self.preprocessing.real_loss(rmse)))
				print "elapsed time: ", time.time() - start_t

				if (e+1)%self.save_every == 0:
					save_name = self.model_path+'model'
					saver.save(sess, save_name, global_step=e+1)
					print "model-%s saved." % (e+1)
			# ============================ for test data ===============================
			print('test for test data...')
			x_test = test_data['x']
			y_test = test_data['y']
			t_loss = 0
			for i in range(len(y_test)):
				if self.cpt_ext:
					feed_dict = {self.model.x_c: np.array(x_test[i][0]), self.model.x_p: np.array(x_test[i][1]), self.model.x_t: np.array(x_test[i][2]), 
								self.model.x_ext: np.array(x_test[i][3]), 
								self.model.y: np.array(y_test[i])}
				else:
					feed_dict = {self.model.x: x_test[i], self.model.y: y_test[i]}
				_, l = sess.run([y_, loss], feed_dict=feed_dict)
				t_loss += l

			# y_val : [batches, batch_size, seq_length, row, col, channel]
			print(np.array(y_test).shape)
			t_count = 0
			for t in range(len(y_test)):
				#print(np.array(y_val[v]).shape)
				t_count += np.prod(np.array(y_test[t]).shape)
			rmse = np.sqrt(t_loss/t_count)
			#rmse = np.sqrt(val_loss/(np.prod(np.array(y_val).shape)))
			print("at epoch " + str(e) + ", test loss is " + str(t_loss) + ' , ' + str(rmse) + ' , ' + str(self.preprocessing.real_loss(rmse)))
			# ============================= for test 1_to_n ==============================
			if self.cpt_ext:
				print('test for next n steps...')
				seq = test_1_to_n_data['data']
				timestamps = test_1_to_n_data['timestamps']
				pre_index = max(self.model.input_conf[0][0]*1, self.model.input_conf[1][0]*24, self.model.input_conf[2][0]*24*7)
				n = 10
				close = 3
				period = 4
				trend = 4
				#start_t = time.time()
				t_loss = 0
				i = pre_index
				while i<len(seq)-n:
					# seq_i : pre_index+n
					seq_i = seq[i-pre_index: i+n]
					time_i = timestamps[i-pre_index: i+n]
					loss_i = 0
					for n_i in range(n):
						x, y = batch_data_cpt_ext(data=seq_i[n_i: n_i+pre_index+1], timestamps=timestamps[n_i: n_i+pre_index+1], 
											batch_size=1, close=close, period=period, trend=trend)
						feed_dict = {self.model.x_c: np.array(x[0][0]), self.model.x_p: np.array(x[0][1]), self.model.x_t: np.array(x[0][2]), 
									self.model.x_ext: np.array(x[0][3]), 
									self.model.y: np.array(y[0])}
						y_p, l = sess.run([y_, loss], feed_dict=feed_dict)
						seq_i[n_i+pre_index] = y_p
						loss_i += l
					y_pred_all.append(seq_i[pre_index:])
					t_loss += loss_i
					i += 1
				row, col, flow = np.array(seq).shape[1:]
				print(row,col,flow)
				test_count = (len(seq)-pre_index-n)*n*(row*col*flow)
				print(test_count)
				rmse = np.sqrt(t_loss/test_count)
				print("test loss is " + str(t_loss) + ' , ' + str(rmse) + ' , ' + str(self.preprocessing.real_loss(rmse)))
				#print("elapsed time: ", time.time() - start_t)
				# if save_outputs:
				# 	np.save('test_n_outputs.npy',y_pred_all)


	def test(self, data, save_outputs=True):
		#x = np.asarray(data['x'])
		#y = np.asarray(data['y'])
		x = data['x']
		y = data['y']

		# build graphs
		y_, loss = self.model.build_model()

		#y = np.asarray(y)
		#y_pred_all = np.ndarray(np.array(y).shape)
		y_pred_all = []
		
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
				if self.cpt_ext:
					feed_dict = {self.model.x_c: np.array(x[i][0]), self.model.x_p: np.array(x[i][1]), self.model.x_t: np.array(x[i][2]), 
									self.model.x_ext: np.array(x[i][3]), 
									self.model.y: np.array(y[i])}
				else:
					feed_dict = {self.model.x: np.array(x[i]), self.model.y: np.array(y[i])}
				y_p, l = sess.run([y_, loss], feed_dict=feed_dict)
				if len(y_pred_all)==0:
					y_pred_all = np.vstack(y_p)
				else:
					y_pred_all = np.vstack((y_pred_all, np.vstack(y_p)))
				t_loss += l
				
			# y : [batches, batch_size, seq_length, row, col, channel]
			print(np.array(y).shape)
			test_count = 0
			for t in range(len(y)):
				#print(np.array(y[t]).shape)
				test_count += np.prod(np.array(y[t]).shape)
			rmse = np.sqrt(t_loss/test_count)
			#rmse = np.sqrt(t_loss/(np.prod(np.array(y).shape)))
			print("test loss is " + str(self.preprocessing.real_loss(rmse)))
			print("elapsed time: ", time.time() - start_t)
			if save_outputs:
				np.save('test_outputs.npy',y_pred_all)

	def test_1_to_n(self, data, n=10, close=3, period=4, trend=4, save_outputs=True):
		seq = data['data']
		timestamps = data['timestamps']
		pre_index = max(close*1, period*24, trend*24*7)
		# build graphs
		y_, loss = self.model.build_model()
		y_pred_all = []
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, self.test_model)
			start_t = time.time()
			#y_pred_all = np.ndarray(y.shape)
			t_loss = 0
			i = pre_index
			while i<len(seq)-n:
				# seq_i : pre_index+n
				seq_i = seq[i-pre_index: i+n]
				time_i = timestamps[i-pre_index: i+n]
				loss_i = 0
				for n_i in range(n):
					x, y = batch_data_cpt_ext(data=seq_i[n_i: n_i+pre_index+1], timestamps=timestamps[n_i: n_i+pre_index+1], 
										batch_size=1, close=close, period=period, trend=trend)
					#print(np.array(x[0][0]).shape)
					#print(np.array(x[0][1]).shape)
					#print(np.array(x[0][2]).shape)
					#print(np.array(x[0][3]).shape)
					#print(np.array(y[0]).shape)
					feed_dict = {self.model.x_c: np.array(x[0][0]), self.model.x_p: np.array(x[0][1]), self.model.x_t: np.array(x[0][2]), 
								self.model.x_ext: np.array(x[0][3]), 
								self.model.y: np.array(y[0])}
					y_p, l = sess.run([y_, loss], feed_dict=feed_dict)
					seq_i[n_i+pre_index] = y_p
					loss_i += l
				y_pred_all.append(seq_i[pre_index:])
				t_loss += loss_i
				i += 1
			row, col, flow = np.array(seq).shape[1:]
			#print(row,col,flow)
			test_count = (len(seq)-pre_index-n)*n*(row*col*flow)
			#print(test_count)
			rmse = np.sqrt(t_loss/test_count)
			print("test loss is " + str(self.preprocessing.real_loss(rmse)))
			print("elapsed time: ", time.time() - start_t)
			if save_outputs:
				np.save('test_n_outputs.npy',y_pred_all)












