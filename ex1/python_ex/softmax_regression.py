import numpy as np
import math

def predict(theta, X):
	return np.exp(X.dot(theta.T))/np.sum([np.exp(X.dot(thetaj.T)) for thetaj in theta])

def objective_func(theta, X, y):
	for i in range(X.shape[0]):
		for k in range(theta.shape[0]):
			(k == y[i])*np.log(theta[k], predict(X[i:,:]))
	h = predict(theta, X)
	print h.shape, y.shape
	return -np.sum(y.flatten()*np.log(h) + (1.-y.flatten())*np.log(1. -h))
	
def gradient(theta, X, y):
	pred = predict(X, y)
	return np.array([(X[:,j]*(pred - y.flatten())).sum() for j in range(theta.shape[0])])