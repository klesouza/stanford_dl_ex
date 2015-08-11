import numpy as np
def linear_regression(theta, X, y):
	m = X.shape[1]
	n = X.shape[0]
	
	f = 0
	g = np.zeros(len(theta))
	
	for i in X:
		f += (theta*i - y)**2