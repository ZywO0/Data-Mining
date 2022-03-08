#--
# linreg0.py
# basic linear regression method applied to an existing data set
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

DEBUGGING = True
DATA_DIR  = '../data/'
DATA_FILE = 'london-borough-profiles-jan2018.csv'
PLOT_DIR  = '../plots/'
LEARNING_RATE = 0.0001
ERROR_MARGIN  = 0.1


#--
# compute_error()
# This function computes the sum of squared errors for the model.
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
    error = error / M
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
#  r2 (scalar)
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

#-get data from a file
try:
# open data file in csv format
    f = open( DATA_DIR + DATA_FILE, encoding='unicode_escape' )
# read contents of data file into "rawdata" list
    rawdata0 = csv.reader( f )
# parse data in csv format
    rawdata = [rec for rec in rawdata0]
# handle exceptions:
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
except Exception as x:
    print('there was an error: ' + str( x ))
    sys.exit()

#-save header and delete from rest of data array
header = rawdata[0]
del rawdata[0]

#-(optionally) print some info about the data set
if DEBUGGING:
    print( 'number of fields = %d' % len( header ) )
    print('fields:')
    i = 0
    for field in header:
        print( 'i=%d field=[%s]' % ( i, field ))
        i = i + 1
    print( 'number of records = %d' % len( rawdata ))

#-save variables of interest:
# column 70 = Male life expectancy, (2012-14)
# column 71 = Female life expectancy, (2012-14)
x = []
y = []
for rec in rawdata:
    err = 0
    tmp1 = rec[70].strip().replace(',','')
    tmp2 = rec[71].strip().replace(',','')
    try:
        f1 = float( tmp1 )
        f2 = float( tmp2 )
    except ValueError as iox:
        err = err + 1
    if ( err == 0 ):
        x.append( f1 )
        y.append( f2 )
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
plt.xlabel( 'age (men)' , fontsize=14 )
plt.ylabel( 'age (women)' , fontsize=14 )
plt.title( 'raw data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'linreg0-raw.png' )
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
plt.xlabel( 'age (men)' , fontsize=14 )
plt.ylabel( 'age (women)' , fontsize=14 )
plt.title( 'partitioned data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'linreg0-partitioned.png' )
plt.show()
plt.close()

#-run gradient descent to compute the regression equation
alpha   = LEARNING_RATE
epsilon = ERROR_MARGIN
# initialise predictions
y_hat = [0 for i in range( M_train )]
# initialise weights
w = [random.random() for i in range( 2 )]
# compute initial error
prev_error = compute_error( M_train, x_train, w, y_train )
for num_iters in range( 100 ):
    # adjust weights using gradient descent
    print(w)
    w = gradient_descent_2( M_train, x_train, w, y_train, alpha )
    # compute error
    curr_error = compute_error( M_train, x_train, w, y_train )
    r2 = compute_r2( M_train, x_train, w, y_train )
    num_iters = num_iters + 1
    print( 'num_iters = %d  prev_error = %f  curr_error = %f  r^2 = %f' % ( num_iters, prev_error, curr_error, r2 ))
    # plot results, when error difference is > 1
    if ( math.fabs( prev_error - curr_error ) > 1 ):
        for j in range( M_train ):
            y_hat[j] = w[0] + w[1] * x_train[j]
        plt.figure()
        plt.plot( x_train, y_train, 'bo', markersize=10 )
        #plt.hold( True )
        plt.plot( x_train, y_hat, 'k', linewidth=3 )
        # set plot axis limits so it all displays nicely
        plt.xlim(( xmin, xmax ))
        plt.ylim(( ymin, ymax ))
        # add plot labels and ticks
        plt.xticks( fontsize=14 )
        plt.yticks( fontsize=14 )
        #plt.title( 'iteration ' + str( num_iters ) + ': y = ' + str( w[0] ) + ' + ' + str( w[1] ) + 'x, r^2=' + str( r2 ))
        print( 'iteration ' + str( num_iters ) + ': y = ' + str( w[0] ) + ' + ' + str( w[1] ) + 'x, error=' + str( curr_error) + ' r^2=' + str( r2 ))
        # save plot
        plt.savefig( PLOT_DIR + 'linreg0-' + str( num_iters ) + '.png' )
        plt.show()
        plt.close()
    # iterate until error hasn't changed much from previous iteration
    if ( prev_error - curr_error < epsilon ):
        converged = True
    else:
        prev_error = curr_error

#-plot and save final regression line from training
plt.figure()
plt.plot( x_train, y_train, 'bo', markersize=10 )
#plt.hold( True )
plt.plot( x_train, y_hat, 'k', linewidth=3 )
# set plot axis limits so it all displays nicely
plt.xlim(( xmin, xmax ))
plt.ylim(( ymin, ymax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
#plt.title( 'regression equation: y = ' + str( w[0] ) + ' + ' + str( w[1] ) + 'x, r^2=' + str( r2 ))
# save plot
plt.savefig( PLOT_DIR + 'linreg0-train.png' )
plt.show()
plt.close()

#-evaluate the model using the test set
test_error = compute_error( M_test, x_test, w, y_test )
test_r2 = compute_r2( M_test, x_test, w, y_test )
print( 'evaluation:  test_error = %f  test_r^2 = %f' % ( test_error, test_r2 ))

#-plot and save regression line from testing
plt.figure()
plt.plot( x_test, y_test, 'rs', markersize=10 )
#plt.hold( True )
y_hat = [0 for i in range( M_test )]
for j in range( M_test ):
    y_hat[j] = w[0] + w[1] * x_test[j]
plt.plot( x_test, y_hat, 'k', linewidth=3 )
# set plot axis limits so it all displays nicely
plt.xlim(( xmin, xmax ))
plt.ylim(( ymin, ymax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
#plt.title( 'regression equation: y = ' + str( w[0] ) + ' + ' + str( w[1] ) + 'x, r^2=' + str( r2 ))
# save plot
plt.savefig( PLOT_DIR + 'linreg0-test.png' )
plt.show()
plt.close()
