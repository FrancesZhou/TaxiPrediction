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

test_num = 10*24

pre_process = MinMaxNormalization01()
print('load data...')
data, all_timestamps_string = load_h5data('./data/NYC14_M16x8_T60_NewEnd.h5')
timestamps2014 = gen_timestamps_for_year_ymdh('2014')
all_timestamps_string = timestamps2014[2160:6552]
all_timestamps_struct = [time.strptime(t, '%Y%m%d%H') for t in all_timestamps_string]
timestamps = [datetime.fromtimestamp(mktime(t)) for t in all_timestamps_struct]
# data: [num, row, col, channel]
print('preprocess train data...')
data = pre_process.fit_transform(data)
data = np.reshape(data, (data.shape[0], -1))

train_data = data[:-test_num]
train_timestamps = timestamps[:-test_num]
test_data = data[-test_num:]
test_timestamps = timestamps[-test_num:]
column_name = [str(e) for e in range(1,train_data.shape[1]+1)]
train_df = pd.DataFrame(train_data, columns=column_name)
train_df.index = pd.DatetimeIndex(train_timestamps)
model_var = VAR(train_df)
results = model_var.fit(10)
#results.summary()
#model_var.select_order(15)
#results = model_var.fit(maxlags=15, ic='aic')
lag_order = results.k_ar
test_data_preindex = np.vstack((train_data[-lag_order:],test_data))
test_predict = np.zeros(test_data.shape)
for i in range(test_num):
    test_predict[i] = results.forecast(test_data_preindex[i:i+lag_order], 1)
n_rmse = np.sqrt(np.sum(np.square(test_predict - test_data))*1.0/np.prod(test_data.shape))
rmse = pre_process.real_loss(n_rmse)
print('test loss is ' + str(n_rmse) + ' , ' + str(rmse))

