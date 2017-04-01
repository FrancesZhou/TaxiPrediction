import numpy as np
import cPickle as pickle
import scipy.io as sio
import h5py
import time
import os

def load_data(filename, split):
	if len(filename)==2:
		d1 = sio.loadmat(filename[0])['p_map']
		d2 = sio.loadmat(filename[1])['d_map']
		data = np.concatenate((d1[:,:,:,np.newaxis], d2[:,:,:,np.newaxis]), axis=3)
	train = data[0:split[0],:,:,:]
	validate = data[split[0]:split[0]+split[1],:,:,:]
	test = data[split[0]+split[1]:split[0]+split[1]+split[2],:,:,:]
	
	return train, validate, test

def load_h5data(fname):
	f = h5py.File(fname, 'r')
	data = f['data'].value
	data = np.asarray(data)
	#data = np.transpose(np.asarray(data), (0,2,3,1))
	timestamps = f['date'].value
	f.close()
	return data, timestamps

def batch_data(data, batch_size=32, input_steps=10, output_steps=10):
	# data: [num, row, col, channel]
	num = data.shape[0]
	# x: [batches, batch_size, input_steps, row, col, channel]
	# y: [batches, batch_size, output_steps, row, col, channel]
	x = []
	y = []
	i = 0
	while i<num-batch_size-input_steps-output_steps:
		batch_x = []
		batch_y = []
		for s in range(batch_size):
			batch_x.append(data[i+s:i+s+input_steps, :, :, :])
			batch_y.append(data[i+s+input_steps:i+s+input_steps+output_steps, :, :, :])
		x.append(batch_x)
		y.append(batch_y)
		i += batch_size
	return x, y
# x: [batches, batch_size, 4]
# y: [batches, batch_size, 1]
# while i<num:
# 	x_b = []
# 	y_b = []
# 	for b in range(batch_size):
# 		x_ = []
# 		if i+b >= num:
# 			break
# 		for d in range(len(depends)):
# 			x_.append(data[i+b-np.array(depends[d]), :, :, :])
# 		x_.append(ext[i])
# 		y_b.append(data[i+b, :, :, :]) 
# 		x_b.append(x_)
# 	x.append(x_b)
# 	y.append(y_b)
# 	i += batch_size

def batch_data_cpt_ext(data, timestamps, batch_size=32, close=3, period=4, trend=4):
	# data: [num, row, col, channel]
	num = data.shape[0]
	flow = data.shape[-1]
	# x: [batches, 
	#[
	#[batch_size, row, col, close*flow], 
	#[batch_size, row, col, period*flow], 
	#[batch_size, row, col, trend*flow],
	#[batch_size, external_dim]
	#]
	#]
	c = 1
	p = 24
	t = 24*7
	depends = [ [c*j for j in range(1, close+1)],
				[p*j for j in range(1, period+1)],
				[t*j for j in range(1, trend+1)] ]
	depends = np.asarray(depends)
	i = max(c*close, p*period, t*trend)
	# external feature
	vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]
	ext = []
	for i in vec:
        	v = [0 for _ in range(7)]
        	v[i] = 1
        	if i >= 5: 
			v.append(0)  # weekend
        	else:
			v.append(1)  # weekday
        	ext.append(v)
	ext = np.asarray(ext)
	# ext plus c p t
	# x: [batches, 4, batch_size]
	# y: [batches, batch_size]
	x = []
	y = []
	while i<num:
		x_b = np.empty(len(depends)+1, dtype=object)
		#y_b = []
		for d in range(len(depends)):
			x_ = []
			for b in range(batch_size):
				if i+b >= num:
					break
				x_.append(np.transpose(np.vstack(data[i+b-np.array(depends[d]), :, :, :]), [1,2,0]))
			x_ = np.array(x_)
			x_b[d] = x_
			#x_b.append(x_)
		# external features
		x_b[-1] = ext[i:min(i+batch_size, num)]
		# y
		y_b = np.transpose(data[i:min(i+batch_size, num), :, :, :],[0,2,3,1])
		x.append(x_b)
		y.append(y_b)
		i += batch_size
	return x, y

def shuffle_batch_data(data, batch_size=32, input_steps=10, output_steps=10):
	num = data.shape[0]
	# shuffle
	data = data[np.random.shuffle(np.arange(num)), :, :, :]

	x = []
	y = []
	i = 0
	while i<num-batch_size-input_steps-output_steps:
		batch_x = []
		batch_y = []
		for s in range(batch_size):
			batch_x.append(data[i+s:i+s+input_steps, :, :, :])
			batch_y.append(data[i+s+input_steps:i+s+input_steps+output_steps, :, :, :])
		x.append(batch_x)
		y.append(batch_y)
		i += batch_size
	return x, y


# show pick-up map and drop-off map
#def imshow_map():



def load_pickle(path):
	with open(path, 'rb') as f:
		file = pickle.load(f)
		print('Loaded %s..' %path)
		return file

def save_pickle(path):
	with open(path, 'rb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		print('Saved %s..' %path)
