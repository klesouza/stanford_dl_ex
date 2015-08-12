import numpy as np
import math

def predict(theta, X):
	return 1./(1. + np.exp(X.dot(theta.T)))

def objective_func(theta, X, y):
	h = predict(theta, X)
	print h.shape, y.shape
	return -np.sum(y.flatten()*np.log(h) + (1.-y.flatten())*np.log(1. -h))
	
def gradient(theta, X, y):
	pred = predict(X, y)
	return np.array([(X[:,j]*(pred - y.flatten())).sum() for j in range(theta.shape[0])])

def minimize(theta, X, y):
	for i in range(10):
		theta = theta - gradient(theta, X, y)
	return theta