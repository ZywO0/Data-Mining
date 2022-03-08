#--
# linreg2.py
# scikit-learn linear regression method applied to a synthetic data set
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
import sklearn.model_selection as model_select
import sklearn.datasets as data
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

DEBUGGING = True
PLOT_DIR  = '../plots/'


#--
# MAIN
#--

#-generate synthetic data for regression
num_features = 1
x, y, p = data.make_regression( n_samples=100, n_features=num_features, n_informative=1, noise=10, coef=True )

#-(optionally) print some info about the data set
if DEBUGGING:
    print( 'number of features = %d' % ( num_features ))

# set number of instances
M = len( x )
# find and scale smallest and largest values of x and y, for plot axis
# limits (so that plots displayed during iterative search have
# consistent axis limits)
xmin = 0.95 * min( x )
xmax = 1.05 * max( x )
ymin = 0.95 * min( y )
ymax = 1.05 * max( y )

#-plot raw data --- always a good idea to do this!
plt.figure()
# plot data points
plt.plot( x, y, 'bo', markersize=10 )
# set plot axis limits so it all displays nicely
plt.xlim(( xmin, xmax ))
plt.ylim(( ymin, ymax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.xlabel( 'x' , fontsize=14 )
plt.ylabel( 'y' , fontsize=14 )
plt.title( 'raw data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'linreg2-raw.png' )
plt.show()
plt.close()

#-partition the data
x_train, x_test, y_train, y_test = model_select.train_test_split( x, y, test_size=0.10 )
M_train = len( x_train )
M_test = len( x_test )

#-plot partitioned data
plt.figure()
plt.plot( x_train, y_train, 'bo', markersize=10, label='train' )
plt.plot( x_test, y_test, 'rs', markersize=10, label='test' )
plt.legend()
# set plot axis limits so it all displays nicely
plt.xlim(( xmin, xmax ))
plt.ylim(( ymin, ymax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.xlabel( 'x' , fontsize=14 )
plt.ylabel( 'y' , fontsize=14 )
plt.title( 'partitioned data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'linreg2-partitioned.png' )
plt.show()
plt.close()

#-use scikit-learn's linear regression model, for comparison
lr = linear_model.LinearRegression() # initialise model
lr.fit( x_train, y_train ) # fit model
y_hat = lr.predict( x_train ) # run prediction on training set
print('scikit regression equation: y = ' + str(lr.intercept_) + ' + ' + str(lr.coef_[0]) + 'x')
print('scikit r2 = ' + str(metrics.r2_score( y_train, y_hat )))
print('scikit error = ' + str(metrics.mean_squared_error( y_train, y_hat )))

#-plot scikit-learn results on training data
plt.figure()
plt.plot( x_train, y_train, 'bo', markersize=10 )
plt.plot( x_train, y_hat, 'k', linewidth=3 )
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
#plt.title( 'scikit regression solution' )
plt.savefig( PLOT_DIR + 'linreg2-scikit-train.png' )
plt.show()
plt.close()

#-compute and plot scikit-learn results on test data
y_hat = lr.predict( x_test ) # run prediction on training set
print('evaluation:')
print('scikit r2 = ' + str(metrics.r2_score( y_test, y_hat )))
print('scikit error = ' + str(metrics.mean_squared_error( y_test, y_hat )))

#-plot scikit-learn results
plt.figure()
plt.plot( x_test, y_test, 'rs', markersize=10 )
plt.plot( x_test, y_hat, 'k', linewidth=3 )
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.savefig( PLOT_DIR + 'linreg2-scikit-test.png' )
plt.show()
plt.close()
