import numpy as np


tar = np.load('test_target.npy')[:2184]
pre = np.load('test_prediction.npy')[:2184]

output_steps = 10
# ================ step-wise rmse =================
# tar: (4358, 10, 16, 16, 2)
# tar = np.reshape(tar, (tar.shape[0], tar.shape[1], -1))
# pre = np.reshape(pre, (pre.shape[0], pre.shape[1], -1))
# tar: (10, 4358, 16, 16, 2) -> (10, 4358*16*16*2)
tar1 = np.reshape(np.transpose(tar,(1,0,2,3,4)), (output_steps, -1))
pre1 = np.reshape(np.transpose(pre,(1,0,2,3,4)), (output_steps, -1))

print(tar1.shape)
#print(pre1.shape)
step_wise_rmse_test = np.sqrt(np.sum(np.square(pre1 - tar1), axis=1)*1.0/tar1.shape[-1])
step_wise_rmse_test = step_wise_rmse_test
print(step_wise_rmse_test)
np.save('step-wise-rmse.npy', step_wise_rmse_test)

# ================ peak/off-peak time rmse ================
# peak time: 8:00 and 9:00
# off-peak time: 15:00 and 16:00
tar2 = np.reshape(tar, (tar.shape[0], -1))
pre2 = np.reshape(pre, (pre.shape[0], -1))
print(tar2.shape)
#print(pre2.shape)
# tar: (4358, 10*16*16*2)
time_rmse = np.zeros(24)
index = np.arange(tar.shape[0]/24)
#print(len(index))
for i in range(24):
	i_index = 24*index+i
	time_rmse[i] = np.sqrt(np.sum(np.square(tar2[i_index] - pre2[i_index]))*1.0/(len(index)*tar2.shape[-1]))
time_rmse = time_rmse
print(time_rmse)
np.save('time_rmse.npy', time_rmse)
