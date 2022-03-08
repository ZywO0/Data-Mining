#--
# per.py
# scikit-learn perceptron classification applied to a synthetic data set
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
LEARNING_RATE = 0.0001


#--
# MAIN
#--

#-generate synthetic data for classification
num_features = 1
X, y = data.make_classification( n_samples=100, n_features=num_features, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1 )

#-(optionally) print some info about the data set
if DEBUGGING:
    print( 'number of features = %d' % ( num_features ))
    print('X shape = ' + str(np.shape( X )))
    print('y shape = ' + str(np.shape( y )))

# set number of instances
M = len( X )
# find and scale smallest and largest values of x and y, for plot axis
# limits (so that plots displayed during iterative search have
# consistent axis limits)
Xmin = 0.95 * min( X )
Xmax = 1.05 * max( X )

#-plot raw data --- always a good idea to do this!
plt.figure()
# plot data points
for j in range( M ):
    if ( y[j] == 0 ):
        h0, = plt.plot( X[j], X[j], 'b.', markersize=10 )
    else:
        h1, = plt.plot( X[j], X[j], 'r+', markersize=10 )
plt.legend(( h0, h1 ), ( 'class0', 'class1' ), loc='upper left' )
# set plot axis limits so it all displays nicely
plt.xlim(( Xmin, Xmax ))
plt.ylim(( Xmin, Xmax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.xlabel( 'X' , fontsize=14 )
plt.ylabel( 'X' , fontsize=14 )
plt.title( 'raw data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'per-raw.png' )
plt.show()
plt.close()

#-partition the data
X_train, X_test, y_train, y_test = model_select.train_test_split( X, y, test_size=0.10 )
M_train = len( X_train )
M_test = len( X_test )

#-plot partitioned data
plt.figure()
# plot data points
for j in range( len( X_train )):
    if ( y_train[j] == 0 ):
        h0train, = plt.plot( X_train[j], X_train[j], 'b.', markersize=10 )
    else:
        h1train, = plt.plot( X_train[j], X_train[j], 'r+', markersize=10 )
for j in range( len( X_test )):
    if ( y_test[j] == 0 ):
        h0test, = plt.plot( X_test[j], X_test[j], 'b>', markersize=10 )
    else:
        h1test, = plt.plot( X_test[j], X_test[j], 'rs', markersize=10 )
plt.legend(( h0train, h1train, h0test, h1test ), ( 'train, class0', 'train, class1', 'test, class0', 'test, class1' ), loc='upper left' )
# set plot axis limits so it all displays nicely
plt.xlim(( Xmin, Xmax ))
plt.ylim(( Xmin, Xmax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.xlabel( 'X' , fontsize=14 )
plt.ylabel( 'X' , fontsize=14 )
plt.title( 'partitioned data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'per-partitioned.png' )
plt.show()
plt.close()

#-use scikit-learn's perceptron model
per = linear_model.Perceptron() # initialise model
per.fit( X_train, y_train ) # fit model to data
print('perceptron weights:')
print('w0 = {}, w1 = {}'.format(per.intercept_, per.coef_))

#-run prediction with model on training data
y_hat = per.predict( X_train ) 
print('training accuracy = ', ( metrics.accuracy_score( y_train, y_hat, normalize=True )))
# plot results 
plt.figure()
# plot raw data points
for j in range( M_train ):
    if ( y_train[j] == 0 ):
        h0train, = plt.plot( X_train[j], X_train[j], 'b.', markersize=10 )
    else:
        h1train, = plt.plot( X_train[j], X_train[j], 'r+', markersize=10 )
plt.legend(( h0train, h1train ), ( 'train, class0', 'train, class1' ), loc='upper left' )
# plot boundary
[xmin,xmax] = plt.xlim()
xx = []
yy = []
for j in range( M_train ):
    xx.append( X_train[j] )
    y_hat = per.intercept_ + X_train[j] * per.coef_[0,0]
    yy.append( y_hat )
plt.plot( xx, yy, 'k-' )
# save plot
plt.savefig( PLOT_DIR + 'per-boundary-train.png' )
plt.show()
plt.close()

#-run prediction with model on test data
y_hat = per.predict( X_test ) 
print('test accuracy =', ( metrics.accuracy_score( y_test, y_hat, normalize=True )))
# plot results
plt.figure()
# plot raw data points
for j in range( M_test ):
    if ( y_test[j] == 0 ):
        h0test, = plt.plot( X_test[j], X_test[j], 'b>', markersize=10 )
    else:
        h1test, = plt.plot( X_test[j], X_test[j], 'rs', markersize=10 )
plt.legend(( h0test, h1test ), ( 'test, class0', 'test, class1' ), loc='upper left' )
#plt.title( 'decision boundary' )
# plot boundary
[xmin,xmax] = plt.xlim()
xx = []
yy = []
for j in range( M_test ):
    xx.append( X_test[j] )
    y_hat = per.intercept_ + X_test[j] * per.coef_[0,0]
    yy.append( y_hat )
plt.plot( xx, yy, 'k-' )
# save plot
plt.savefig( PLOT_DIR + 'per-boundary-test.png' )
plt.show()
plt.close()
