import numpy as np

tar = np.load('test_target.npy')
tar = np.float32(tar)
tar = tar*23432.0
np.save('test_target.npy', tar)

pre = np.load('test_prediction.npy')
pre = np.float32(pre)
pre = pre*23432.0
np.save('test_prediction.npy', pre)
