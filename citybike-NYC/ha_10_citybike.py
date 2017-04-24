from __future__ import print_function
import numpy as np
import sys
# for mac debug
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/model/')
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/util/')
# for server running
sys.path.append('/home/zx/TaxiPrediction/model/')
sys.path.append('../util/')
sys.path.append('../data/')
from utils import *
from preprocessing import *

input_steps = 10
output_steps = 10
print('load train, validate, test data...')
split = [17520, 4416, 4368]
data, train_data, val_data, test_data = load_npy_data(filename=['../data/citybike/p_map.npy', '../data/citybike/d_map.npy'], split=split)

s = data.shape
p = 24*7
all_mean = np.zeros((p, s[1], s[2], s[3]))
index = np.arange(split[0]/p)
for i in range(p):
	#index = np.arange(split[0]/p)
    d = data[p*index+i]
    all_mean[i] = np.mean(d, axis=0)

val_real = []
val_predict = []
for i in range(val_data.shape[0] - output_steps):
	val_index = np.arange(split[0]+i, split[0]+i+output_steps)
	val_real.append(val_data[i: i+output_steps])
	val_p = np.remainder(val_index, p)
	val_predict.append(all_mean[val_p])
val_real = np.array(val_real)
val_predict = np.array(val_predict)
# for test data
test_real = []
test_predict = []
for i in range(test_data.shape[0] - output_steps):
	test_index = np.arange(split[0]+split[1]+i, split[0]+split[1]+i+output_steps)
	test_real.append(test_data[i: i+output_steps])
	test_p = np.remainder(test_index, p)
	test_predict.append(all_mean[test_p])
test_real = np.array(test_real)
test_predict = np.array(test_predict)

n_rmse_val = np.sqrt(np.sum(np.square(val_predict - val_real))*1.0/np.prod(val_real.shape))
n_rmse_test = np.sqrt(np.sum(np.square(test_predict - test_real))*1.0/np.prod(test_real.shape))
np.save('../citybike-results/results/HA/test_target.npy', test_real)
np.save('../citybike-results/results/HA/test_prediction.npy', test_predict)
#rmse_val = pre_process.real_loss(n_rmse_val)
#rmse_test = pre_process.real_loss(n_rmse_test)
#print('val loss is ' + str(n_rmse_val) + ' , ' + str(rmse_val))
#print('test loss is ' + str(n_rmse_test) + ' , ' + str(rmse_test))
print('val loss is ' + str(n_rmse_val))
print('test loss is ' + str(n_rmse_test))

