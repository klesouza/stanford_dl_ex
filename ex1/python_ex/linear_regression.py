import numpy as np 

def predict(theta, X):
	return X.dot(theta.T)
	
def objective_func(theta, X, y):
	pred = predict(theta, X)
	J = ((pred - y.flatten())**2).sum()/2
	return J

def gradient(theta, X, y):
	pred = predict(theta, X)
	return np.array([(X[:,j]*(pred - y.flatten())).sum() for j in range(theta.shape[0])])

def linear_regression(theta, X, y):
	f = objective_func(theta, X, y)
	g = gradient(theta, X, y)	
	return (f, g)

def minimize(theta, X, y):
	for i in range(10):
		theta = theta - gradient(theta, X, y)
		print theta.shape
	return theta