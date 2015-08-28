import numpy as np

def check(cost_func, grad, theta, X, y, checks = 10, epsilon = 0.00001):
	T = theta.copy()
	sum_error = 0
	for i in range(checks):
		np.random.shuffle(T)
		f0 = cost_func(T-epsilon, X, y)
		f1 = cost_func(T+epsilon, X, y)
		g = grad(T, X, y)
		g_est = (f1-f0)/(2*epsilon)
		error = abs(g-g_est)
		sum_error += error
		print "check: {0}; error: {1}".format(i,error)
	avg = sum_error/checks
	print avg