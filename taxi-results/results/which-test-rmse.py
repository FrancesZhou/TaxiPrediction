import numpy as np

methods = ['HA', 'ARMA', 'VAR', 'ResNet', 'ConvLSTM', 'AttConvLSTM']

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
	error = tar[:4344] - pre[:4344]
	rmse = np.sqrt(np.sum(np.square(error))/np.prod(error.shape))
	variance = np.sqrt(np.var(error))
	#rmse = np.sqrt(np.sum(np.square(tar-pre))/np.prod(tar.shape))
	#variance = np.sqrt(np.var(tar-pre))
	#max_abs_error = np.max(np.absolute((tar-pre))/tar)
	#print('rmse: %.4f, max_abs_error: %.4f, variance: %.4f' %(rmse, max_abs_error, variance))
	print('rmse: %.4f, variance: %.4f' %(rmse, variance))

# for i in range(2,len(methods)):
# 	print methods[i]
# 	tar_file = methods[i]+'/test_target.npy'
# 	#print tar_file
# 	tar = np.load(tar_file)
# 	pre = np.load(methods[i]+'/test_prediction.npy')
# 	rmse = np.sqrt(np.sum(np.square(tar-pre))/np.prod(tar.shape))*444.0
# 	#max_abs_error = np.max(np.absolute((tar-pre))/tar)
# 	variance = np.sqrt(np.var(tar-pre))*444.0
# 	#print('rmse: %.4f, max_abs_error: %.4f, variance: %.4f' %(rmse, max_abs_error, variance))
# 	print('rmse: %.4f, variance: %.4f' %(rmse, variance))

#step_wise_rmse_test = np.sqrt(np.sum(np.square(pre1 - tar1), axis=1)*1.0/tar1.shape[-1])

