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
sys.path.append('../util/')
sys.path.append('../data/')
from utils import *
from preprocessing import *

input_steps = 10
output_steps = 10
#run_times = 14*24*10
print('load train, validate, test data...')
split = [17424, 4464, 4320]
data, train_data, val_data, test_data = load_npy_data(filename=['../data/citybike_10_minutes/p_map.npy', '../data/citybike_10_minutes/d_map.npy'], split=split)
# data: [num, row, col, channel]
print('preprocess train data...')
all_timestamps_string = gen_timestamps('2016', gen_timestamps_for_year=gen_timestamps_for_year_ymdhm)
all_timestamps_string = all_timestamps_string[:-4416]
all_timestamps_struct = [time.strptime(t, '%Y%m%d%H%M') for t in all_timestamps_string]
timestamps = [datetime.fromtimestamp(mktime(t)) for t in all_timestamps_struct]
# data: [num, row, col, channel]
print('preprocess and get test data...')
data = np.reshape(data, (data.shape[0], -1))
#val_data = data[split[0]:split[0]+split[1]]
test_data = data[split[0]+split[1]:]
data = np.transpose(data)
#val_data = np.transpose(val_data)
test_data = np.transpose(test_data)

print('======================= ARMA for test ===============================')
loss = 0
error_count = 0
run_times = test_data.shape[-1]
index_all = np.zeros(run_times)
error_index = []
test_target = np.zeros([run_times, output_steps])
test_prediction = np.zeros([run_times, output_steps])
# test_data: [feature, num]
for j in range(test_data.shape[-1]-output_steps):
    print('run '+str(j))
    i = np.random.randint(data.shape[0])
    #j = np.random.randint(test_data.shape[-1]-output_steps)
    train_df = pd.DataFrame(data[i][j:split[0]+split[1]+j])
    train_df.index = pd.DatetimeIndex(timestamps[j:split[0]+split[1]+j])
    try:
        results = ARMA(train_df, order=(2,2)).fit(trend='nc', disp=-1)
    except:
        error_index.append(j)
        #error_count += 1
        continue
    pre, _, _ = results.forecast(output_steps)
    test_real = test_data[i][j:j+output_steps]
    index_all[j] = i
    test_target[j] = test_real
    test_prediction[j] = pre
    #loss += np.sum(np.square(pre - test_real))
print('================ calculate rmse for test data ============')
#n_rmse_val = np.sqrt(np.sum(np.square(val_predict - val_real))*1.0/np.prod(val_real.shape))
#n_rmse_test = np.sqrt(np.sum(np.square(test_predict - test_real))*1.0/np.prod(test_real.shape))
#rmse_val = pre_process.real_loss(n_rmse_val)
#rmse_test = pre_process.real_loss(n_rmse_test)
#print('val loss is ' + str(n_rmse_val) + ' , ' + str(rmse_val))
#print('test loss is ' + str(n_rmse_test) + ' , ' + str(rmse_test))
#print('val loss is ' + str(n_rmse_val))
loss = np.sum(np.square(test_prediction-test_target))
error_index = np.asarray(error_index)
error_count = len(error_index)
print('run times: '+str(run_times))
print('error count: '+str(error_count))
rmse = np.sqrt(loss/((run_times-error_count)*output_steps))
print('test loss is ' + str(rmse))
np.save('citybike-10-minutes-results/results/ARMA_4320/test_target.npy', test_target)
np.save('citybike-10-minutes-results/results/ARMA_4320/test_prediction.npy', test_prediction)
np.save('citybike-10-minutes-results/results/ARMA_4320/index_all.npy', index_all)
np.save('citybike-10-minutes-results/results/ARMA_4320/error_index.npy', error_index)

