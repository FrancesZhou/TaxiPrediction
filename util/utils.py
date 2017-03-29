import numpy as np
import cPickle as pickle
import scipy.io as sio
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