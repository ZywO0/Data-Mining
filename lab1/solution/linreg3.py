#--
# linreg3.py
# scikit-learn linear regression method applied to a synthetic data
# set, where the data is split into training and test sets
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import sys
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

DEBUGGING = True
PLOT_DIR  = '../plots/'
LEARNING_RATE = 0.0001
ERROR_MARGIN  = 0.1


#--
# compute_error()
# This function computes the sum-of-squares error for the model.
# inputs:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameters values (of size 2)
#  y = list of target values
# output:
#  error (scalar)
#--
def compute_error( M, x, w, y ):
    error = 0
    y_hat = [0 for i in range( M )]
    for j in range( M ):
        y_hat[j] = w[0] + w[1] * x[j]
        error = error + math.pow(( y[j] - y_hat[j] ), 2 )
    error = error / 2.0
    return( error )


#--
# compute_r2()
# This function computes R^2 for the model.
# inputs:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameters values (of size 2)
#  y = list of target values
# output:
#  error (scalar)
#--
def compute_r2( M, x, w, y ):
    u = 0
    v = 0
    y_hat = [0 for i in range( M )]
    y_mean = np.mean( y )
    for j in range( M ):
        y_hat[j] = w[0] + w[1] * x[j]
        u = u + math.pow(( y[j] - y_hat[j] ), 2 )
        v = v + math.pow(( y[j] - y_mean ), 2 )
    r2 = 1.0 - ( u / v )
    return( r2 )


#--
# gradient_descent_2()
# this function solves linear regression with gradient descent for 2
# parameters.
# inputs:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameter values (of size 2)
#  y = list of target values
#  alpha = learning rate
# output:
#  w = updated list of parameter values
#---
def gradient_descent_2( M, x, w, y, alpha ):
    for j in range( M ):
        # compute prediction for this instance
        y_hat = w[0] + w[1] * x[j]
        # compute prediction error for this instance
        error = y[j] - y_hat
        # adjust by partial error (for this instance)
        w[0] = w[0] + alpha * error * 1    * ( 1.0 / M )
        w[1] = w[1] + alpha * error * x[j] * ( 1.0 / M )
    return w



#--
# MAIN
#--

#-generate synthetic data set
x, y, p = datasets.make_regression( n_samples=1000, n_features=1, n_informative=1, noise=10, coef=True )

#-split data into training and test sets
x_train, x_test, y_train, y_test = model_selection.train_test_split( x, y, test_size=0.10 )

#-use scikit-learn's linear regression model, for comparison
lr = linear_model.LinearRegression()
lr.fit( x_train, y_train )
y_hat = lr.predict( x_test )
print('scikit regression equation: y = ' + str(lr.intercept_) + ' + ' + str(lr.coef_[0]) + 'x')
print('scikit r2 = ' + str(metrics.r2_score( y_test, y_hat )))
print('scikit error = ' + str(metrics.mean_squared_error( y_test, y_hat )))

#-plot scikit-learn results
plt.figure()
plt.scatter( x_test, y_test )
#plt.hold( True )
plt.plot( x_test, y_hat, 'r' )
plt.title( 'scikit regression solution' )
plt.savefig( PLOT_DIR + 'linreg3-scikit-final.png' )
plt.show()
plt.close()
