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
        data = np.array(data, dtype=np.float32)
    train = data[0:split[0],:,:,:]
    validate = data[split[0]:split[0]+split[1],:,:,:]
    test = data[split[0]+split[1]:split[0]+split[1]+split[2],:,:,:]
    return data, train, validate, test

def load_npy_data(filename, split):
    if len(filename)==2:
        d1 = np.load(filename[0])
        d2 = np.load(filename[1])
        data = np.concatenate((d1[:,:,:,np.newaxis], d2[:,:,:,np.newaxis]), axis=3)
        data = np.array(data, dtype=np.float32)
    train = data[0:split[0],:,:,:]
    validate = data[split[0]:split[0]+split[1],:,:,:]
    test = data[split[0]+split[1]:split[0]+split[1]+split[2],:,:,:]
    return data, train, validate, test

def load_h5data(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    data = np.asarray(data)
    data = np.transpose(np.asarray(data), (0,2,3,1))
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
    #flow = data.shape[1]
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
    ext = external_feature(timestamps)
    # ext plus c p t
    # x: [batches, 4, batch_size]
    # y: [batches, batch_size]
    x = []
    y = []
    while i<num:
        x_b = np.empty(len(depends)+1, dtype=object)
        for d in range(len(depends)):
            x_ = []
            for b in range(batch_size):
                if i+b >= num:
                    break
                x_.append(np.transpose(np.vstack(np.transpose(data[i+b-np.array(depends[d]), :, :, :],[0,3,1,2])), [1,2,0]))
            x_ = np.array(x_)
            x_b[d] = x_
            #x_b.append(x_)
        # external features
        x_b[-1] = ext[i:min(i+batch_size, num)]
        # y
        y_b = data[i:min(i+batch_size, num), :, :, :]
        x.append(x_b)
        #print(y_b.shape)
        y.append(y_b)
        i += batch_size
    return x, y

def external_feature(timestamps):
    vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]
    ext = []
    for j in vec:
        v = [0 for _ in range(7)]
        v[j] = 1
        if j >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ext.append(v)
    ext = np.asarray(ext)
    return ext

def gen_timestamps_for_year(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            t = [year+month[m]+day[d]]
            t_d = t*24
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps(years, gen_timestamps_for_year=gen_timestamps_for_year):
    timestamps = []
    for y in years:
        timestamps.append(gen_timestamps_for_year(y))
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps_for_year_ymdh(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hour1 = ['0'+str(e) for e in range(0,10)]
    hour2 = [str(e) for e in range(10,24)]
    hour = hour1+hour2
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            #t = [year+month[m]+day[d]]
            t_d = []
            for h in range(24):
                t_d.append(year+month[m]+day[d]+hour[h])
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps_for_year_ymdhm(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hour1 = ['0'+str(e) for e in range(0,10)]
    hour2 = [str(e) for e in range(10,24)]
    hour = hour1+hour2
    minute = ['00', '10', '20', '30', '40', '50']
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            #t = [year+month[m]+day[d]]
            t_d = []
            for h in range(24):
                a = [year+month[m]+day[d]+hour[h]+e for e in minute]
                #t_d = [t_d.append(year+month[m]+day[d]+hour[h]+e) for e in minute]
                t_d.append(a)
            t_d = np.hstack(np.array(t_d))
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

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

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' %path)
        return file

def save_pickle(path,data):
    with open(path, 'rb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' %path)
