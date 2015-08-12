import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from linear_regression import *
		
data = np.loadtxt('ex1\python_ex\housing.data')
data = np.random.permutation(data)
data = np.insert(data,0,np.ones(data.shape[0]),axis=1)
a = np.array([[1,2,1],[3,4,2],[5,6,1],[7,8,2]])

train_x = data[0:400,0:-1]
train_y = data[0:400,-1:]
test_x = data[400:,0:-1]
test_y = data[400:,-1:]
m = train_x.shape[0]
n = train_x.shape[1]

theta = np.array(np.random.permutation(range(1,n+1)), dtype='float')
import math
sorting = np.argsort(test_y, axis=0)
opt = optimize.minimize(objective_func, theta, args=(train_x,train_y), options={'maxiter':200})
train_error = predict(opt.x,train_x)-train_y
print 'Training error:', math.sqrt(np.mean(train_error**2))

pred_test = predict(opt.x, test_x[sorting])
test_error = pred_test - test_y[sorting]
print 'Test error:', math.sqrt(np.mean(test_error**2))

plt.plot(range(106),pred_test,'rx', range(106),test_y[sorting].flatten(),'g^')
plt.show()