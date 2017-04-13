from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats
from time import mktime
from datetime import datetime
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.vector_ar.var_model import VAR
import sys
# for mac debug
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/model/')
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/util/')
# for server running
sys.path.append('/home/zx/TaxiPrediction/model/')
sys.path.append('./util/')
sys.path.append('./data/')
from utils import *
from preprocessing import *

input_steps = 10
output_steps = 10
run_times = 480
#pre_process = MinMaxNormalization01()
print('load train, validate, test data...')
split = [43824, 8760, 8760]
data, train_data, val_data, test_data = load_data(filename=['data/p_map.mat', 'data/d_map.mat'], split=split)
# data: [num, row, col, channel]
print('preprocess train data...')
#pre_process.fit(train_data)
all_timestamps_string = gen_timestamps(['2009','2010','2011','2012','2013','2014','2015'], gen_timestamps_for_year=gen_timestamps_for_year_ymdh)
all_timestamps_struct = [time.strptime(t, '%Y%m%d%H') for t in all_timestamps_string]
timestamps = [datetime.fromtimestamp(mktime(t)) for t in all_timestamps_struct]
# data: [num, row, col, channel]
print('preprocess train data...')
#data = pre_process.transform(data)
data = np.reshape(data, (data.shape[0], -1))
val_data = data[split[0]:split[0]+split[1]]
test_data = data[split[0]+split[1]:]
data = np.transpose(data)
val_data = np.transpose(val_data)
test_data = np.transpose(test_data)
# data: [64*64*2, num]
#train_data = data[:][:split[0]]

#train_timestamps = timestamps[:split[0]]
# validate and test data
# val_real = np.zeros((data.shape[0], val_data.shape[-1]-output_steps, output_steps))
# val_predict = np.zeros(val_real.shape)
# test_real = np.zeros((data.shape[0], test_data.shape[-1]-output_steps, output_steps))
# test_predict = np.zeros(test_real.shape)
# ARMA for validate data
# print('======================== ARMA for validate ==========================')
# for i in range(data.shape[0]):
#     print('validate, i = '+str(i))
#     for j in range(val_data.shape[-1]-output_steps):
#         train_df = pd.DataFrame(data[i][j:split[0]+j])
#         train_df.index = pd.DatetimeIndex(timestamps[j:split[0]+j])
#         results = ARMA(train_df, order=(2,2)).fit(trend='nc', disp=-1)
#         pre, _, _ = results.forecast(output_steps)
#         val_real[i][j] = val_data[i][j:j+output_steps]
#         val_predict[i][j] = pre
# ARMA for test data
# print('======================= ARMA for test ===============================')
# for i in range(data.shape[0]):
#     print('test, i = '+str(i))
#     for j in range(test_data.shape[-1]-output_steps):
#         train_df = pd.DataFrame(data[i][j:split[0]+split[1]+j])
#         train_df.index = pd.DatetimeIndex(timestamps[j:split[0]+split[1]+j])
#         results = ARMA(train_df, order=(2,2)).fit(trend='nc', disp=-1)
#         pre, _, _ = results.forecast(output_steps)
#         test_real[i][j] = test_data[i][j:j+output_steps]
#         test_predict[i][j] = pre
print('======================= ARMA for test ===============================')
loss = 0
for r in range(run_times):
    i = np.random.randint(data.shape[0])
    j = np.random.randint(test_data.shape[-1]-output_steps)
    train_df = pd.DataFrame(data[i][j:split[0]+split[1]+j])
    train_df.index = pd.DatetimeIndex(timestamps[j:split[0]+split[1]+j])
    results = ARMA(train_df, order=(2,2)).fit(trend='nc', disp=-1)
    pre, _, _ = results.forecast(output_steps)
    test_real = test_data[i][j:j+output_steps]
    loss += np.sum(np.square(pre - test_real))
print('================ calculate rmse for validate and test data ============')
#n_rmse_val = np.sqrt(np.sum(np.square(val_predict - val_real))*1.0/np.prod(val_real.shape))
#n_rmse_test = np.sqrt(np.sum(np.square(test_predict - test_real))*1.0/np.prod(test_real.shape))
#rmse_val = pre_process.real_loss(n_rmse_val)
#rmse_test = pre_process.real_loss(n_rmse_test)
#print('val loss is ' + str(n_rmse_val) + ' , ' + str(rmse_val))
#print('test loss is ' + str(n_rmse_test) + ' , ' + str(rmse_test))
#print('val loss is ' + str(n_rmse_val))
rmse = np.sqrt(loss/(run_times*output_steps))
print('test loss is ' + str(n_rmse_test))

