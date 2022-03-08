#--
# data2.py
# data exploration using synthetic data for classification
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

#-generate synthetic data for classification
num_features = 1
x, y = data.make_classification( n_samples=100, n_features=num_features, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1 )

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
ymin = 0.95 * min( x )
ymax = 1.05 * max( x )

#-plot raw data --- always a good idea to do this!
plt.figure()
# plot data points
for j in range( M ):
    if ( y[j] == 0 ):
        h0, = plt.plot( x[j], x[j], 'b.', markersize=10 )
    else:
        h1, = plt.plot( x[j], x[j], 'r+', markersize=10 )
plt.legend(( h0, h1 ), ( 'class0', 'class1' ))
# set plot axis limits so it all displays nicely
plt.xlim(( xmin, xmax ))
plt.ylim(( ymin, ymax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.xlabel( 'x' , fontsize=14 )
plt.ylabel( 'x' , fontsize=14 )
plt.title( 'raw data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'data2-raw.png' )
plt.show()
plt.close()

#-partition the data
x_train, x_test, y_train, y_test = model_select.train_test_split( x, y, test_size=0.10 )

#-plot partitioned data
plt.figure()
# plot data points
for j in range( len( x_train )):
    if ( y_train[j] == 0 ):
        h0train, = plt.plot( x_train[j], x_train[j], 'b.', markersize=10 )
    else:
        h1train, = plt.plot( x_train[j], x_train[j], 'r+', markersize=10 )
for j in range( len( x_test )):
    if ( y_test[j] == 0 ):
        h0test, = plt.plot( x_test[j], x_test[j], 'b>', markersize=10 )
    else:
        h1test, = plt.plot( x_test[j], x_test[j], 'rs', markersize=10 )
plt.legend(( h0train, h1train, h0test, h1test ), ( 'train, class0', 'train, class1', 'test, class0', 'test, class1' ))
# set plot axis limits so it all displays nicely
plt.xlim(( xmin, xmax ))
plt.ylim(( ymin, ymax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.xlabel( 'x' , fontsize=14 )
plt.ylabel( 'x' , fontsize=14 )
plt.title( 'partitioned data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'data2-partitioned.png' )
plt.show()
plt.close()
