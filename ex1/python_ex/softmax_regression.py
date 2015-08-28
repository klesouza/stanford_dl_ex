import numpy as np
import math

def predict(theta, X):
	predicted = []
	print theta.shape
	for i in range(X.shape[0]):
		predicted.append([])
		print theta.shape
		for k in range(theta.shape[0]):
			print k
			x = X[i].reshape(1, X[i].shape[0])
			t = theta[k].reshape(1,theta[k].shape[0])
			predicted[i].append(np.exp(x.dot(t.T))/np.exp(x.dot(theta.T)).sum())
	return np.array(predicted)

def objective_func(theta, x, y):
	print theta.shape, x.shape, y.shape
	return np.log(predict(theta, x))[np.arange(x.shape[0], y)].sum()
	

def gradient(theta, X, y):
	pred = predict(theta, X)
	return X.T.dot((pred == pred[np.arange(10,), y].reshape(10,1)).astype(float) - pred).T