from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats
from time import mktime
from datetime import datetime
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
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
pre_process = MinMaxNormalization01()
print('load train, validate, test data...')
split = [43824, 8760, 8760]
data, train_data, val_data, test_data = load_data(filename=['data/p_map.mat', 'data/d_map.mat'], split=split)
# data: [num, row, col, channel]
print('preprocess train data...')
pre_process.fit(train_data)
all_timestamps_string = gen_timestamps(['2009','2010','2011','2012','2013','2014','2015'], gen_timestamps_for_year=gen_timestamps_for_year_ymdh)
all_timestamps_struct = [time.strptime(t, '%Y%m%d%H') for t in all_timestamps_string]
timestamps = [datetime.fromtimestamp(mktime(t)) for t in all_timestamps_struct]
# data: [num, row, col, channel]
print('preprocess train data...')
data = pre_process.transform(data)
data = np.reshape(data, (data.shape[0], -1))

train_data = data[:split[0]]
val_data = data[split[0]:split[0]+split[1]]
test_data = data[split[0]+split[1]:]
train_timestamps = timestamps[:split[0]]
#val_timestamps = timestamps[split[0]:split[0]+split[1]]
#test_timestamps = timestamps[split[0]+split[1]:]

column_name = [str(e) for e in range(1,train_data.shape[1]+1)]
train_df = pd.DataFrame(train_data, columns=column_name)
train_df.index = pd.DatetimeIndex(train_timestamps)
model_var = VAR(train_df)
results = model_var.fit(1)
lag_order = results.k_ar
val_data_preindex = np.vstack((train_data[-lag_order:], val_data))
#val_predict = np.zeros(val_data.shape)
test_data_preindex = np.vstack((val_data[-lag_order:], test_data))
#test_predict = np.zeros(test_data.shape)
# validate and test data
val_real = []
val_predict = []
test_real = []
test_predict = []
for i in range(val_data.shape[0]-output_steps):
    val_real.append(val_data[i: i+output_steps])
    val_predict.append(results.forecast(val_data_preindex[i:i+lag_order], 10))
for i in range(test_data.shape[0]-output_steps):
    test_real.append(test_data[i: i+output_steps])
    test_predict.append(results.forecast(test_data_preindex[i:i+lag_order], 10))
val_real = np.array(val_real)
val_predict = np.array(val_predict)
test_real = np.array(test_real)
test_predict = np.array(test_predict)

n_rmse_val = np.sqrt(np.sum(np.square(val_predict - val_data))*1.0/np.prod(val_real.shape))
n_rmse_test = np.sqrt(np.sum(np.square(test_predict - test_data))*1.0/np.prod(test_real.shape))
rmse_val = pre_process.real_loss(n_rmse_val)
rmse_test = pre_process.real_loss(n_rmse_test)
print('val loss is ' + str(n_rmse_val) + ' , ' + str(rmse_val))
print('test loss is ' + str(n_rmse_test) + ' , ' + str(rmse_test))

