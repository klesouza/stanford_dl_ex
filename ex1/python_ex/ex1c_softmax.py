import numpy as np
from scipy import optimize
from read_mnist import *
from softmax_regression import *
import os
print os.getcwd()
path = 'ex1\\python_ex\\mnist\\'
(test_x,test_y) = get_labeled_data(path+'t10k-images-idx3-ubyte.gz', path+'t10k-labels-idx1-ubyte.gz')
(train_x,train_y) = get_labeled_data(path+'train-images-idx3-ubyte.gz', path+'train-labels-idx1-ubyte.gz')

m = train_x.shape[0]
n = train_x.shape[1]
num_classes = 10
theta = np.random.rand(num_classes-1,n)*0.001
print train_x.shape,train_y.shape
pass
# predict(theta, train_x)
# pass
import math
print theta.shape
opt = optimize.fmin_bfgs(objective_func, theta, gradient, args=(train_x,train_y))
print 'minimized'
print opt.x.shape
train_error = predict(opt.x,train_x)==train_y
print 'Training error:', train_error.sum()/train_y.shape[0]

pred_test = predict(opt.x, test_x)
test_error = pred_test == test_y
print 'Test error:', test_error.sum()/test_y.shape[0]