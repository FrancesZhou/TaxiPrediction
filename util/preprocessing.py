import numpy as np

class MinMaxNormalization01(object):
	def __init__(self, ):
		pass

	def fit(self, data):
		self._min = np.amin(data)
		self._max = np.amax(data)
		print("min: ", self._min, "max:", self._max)

	def transform(self, data):
		norm_data = 1. * (data - self._min) / (self._max - self._min)
		return norm_data

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)

	def inverse_transform(self, data):
		inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		return inverse_norm_data

	def real_loss(self, loss):
		# loss is rmse
		return loss*(self._max - self._min)
		#return real_loss



class MinMaxNormalization_neg_1_pos_1(object):
    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
    	X = (X + 1.)/2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

    def real_loss(self, loss):
    	# loss is rmse
    	return loss*(self._max - self._min)/2.
	#return real_loss
