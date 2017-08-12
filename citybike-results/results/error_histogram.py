import numpy as np

methods = ['HA', 'ARMA_10', 'VAR', 'ResNet', 'ConvLSTM', 'AttConvLSTM']

bins = np.arange(0, 1.1, 0.1)
bins = np.append(bins, float('inf'))
print bins

count = np.zeros((len(methods), len(bins) - 1))

for i in range(len(methods)):
	print methods[i]
	tar_file = methods[i]+'/test_target.npy'
	#print tar_file
	tar = np.load(tar_file)
	pre = np.load(methods[i]+'/test_prediction.npy')
	#print tar.shape
	if i>3:
		tar = np.reshape(tar, (tar.shape[0]*tar.shape[1], -1))
		pre = np.reshape(pre, (pre.shape[0]*pre.shape[1], -1))
	tar = tar[:2184]+1
	pre = pre[:2184]+1
	error = np.reshape(tar - pre, (-1))
	#rmse = np.sqrt(np.sum(np.square(tar-pre))/np.prod(tar.shape))
	rmse = np.sqrt(np.sum(np.square(error))/len(error))
	# ------------ calculate histogram
	h, _ = np.histogram(np.absolute(error)/(np.reshape(tar,(-1))), bins=bins)
	#print('sum_h: %d, sum_error: %d' %(sum(h), len(error)))
	p = h*1.0/len(error)
	count[i] = p
	#variance = np.sqrt(np.var(error))
	#rmse = np.sqrt(np.sum(np.square(tar-pre))/np.prod(tar.shape))
	#max_abs_error = np.max(np.absolute((tar-pre))/tar)
	print('rmse: %.4f' %(rmse))
	print p