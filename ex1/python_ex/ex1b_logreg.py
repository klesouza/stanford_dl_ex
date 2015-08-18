import numpy as np
from scipy import optimize
from read_mnist import *
from logistic_regression import *
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

path = 'mnist/'
(test_x,test_y) = get_labeled_data(os.path.join(path,'t10k-images-idx3-ubyte.gz'), os.path.join(path,'t10k-labels-idx1-ubyte.gz'),[0,1])
(train_x,train_y) = get_labeled_data(path+'train-images-idx3-ubyte.gz', path+'train-labels-idx1-ubyte.gz',[0,1])

m = train_x.shape[0]
n = train_x.shape[1]

theta = np.array(np.random.permutation(range(1,n+1)), dtype='float')*0.001
import math
opt = optimize.minimize(objective_func, theta, args=(train_x,train_y), options={'maxiter':200})
train_error = predict(opt.x,train_x)==train_y
print 'Training error:', train_error.sum()/train_y.shape[0]

pred_test = predict(opt.x, test_x)
test_error = pred_test == test_y
print 'Test error:', test_error.sum()/test_y.shape[0]