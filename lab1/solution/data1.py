#--
# data1.py
# data exploration using synthetic data for regression
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
plt.savefig( PLOT_DIR + 'data1-raw.png' )
plt.show()
plt.close()

#-partition the data
x_train, x_test, y_train, y_test = model_select.train_test_split( x, y, test_size=0.10 )

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
plt.savefig( PLOT_DIR + 'data1-partitioned.png' )
plt.show()
plt.close()
