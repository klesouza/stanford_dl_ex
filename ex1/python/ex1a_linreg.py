import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

def objective_func(theta, X, y):
	pred = X.dot(theta.T)
	J = ((pred - y.flatten())**2).sum()/2
	return J

def gradient(theta, X, y):
	pred = X.dot(theta.T)
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
		
data = np.loadtxt('ex1\python\housing.data')
data = np.random.permutation(data)
#data = np.transpose(data)
data = np.insert(data,0,np.ones(data.shape[0]),axis=1)
a = np.array([[1,2,1],[3,4,2],[5,6,1],[7,8,2]])

train_x = data[0:400,0:-1]
train_y = data[0:400,-1:]
test_x = (data[data[:,-1].argsort()])[400:,0:-1]
test_y = (data[data[:,-1].argsort()])[400:,-1:]
m = train_x.shape[0]
n = train_x.shape[1]

theta = np.array(np.random.permutation(range(1,n+1)), dtype='float')

#print linear_regression(theta, train_x, train_y)
opt = optimize.minimize(objective_func, theta, args=(train_x,train_y), options={'maxiter':200})
plt.plot(range(106),test_x.dot(opt.x.T),'rx', range(106),test_y.flatten(),'g^')
plt.show()
#print minimize(theta, train_x, train_y)