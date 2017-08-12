import numpy as np

tar = np.load('test_target.npy')
tar = tar*23432.0
np.save('test_target.npy', tar)

pre = np.load('test_prediction.npy')
pre = pre*23432.0
np.save('test_prediction.npy', pre)