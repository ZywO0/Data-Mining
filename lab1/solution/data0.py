#--
# data0.py
# data exploration using downloaded data
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


#--
# MAIN
#--

#-get data from a file
try:
# open data file in csv format
    f = open( DATA_DIR + DATA_FILE, encoding='unicode_escape')
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
plt.savefig( PLOT_DIR + 'data0-raw.png' )
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
plt.xlabel( 'age (men)' , fontsize=14 )
plt.ylabel( 'age (women)' , fontsize=14 )
plt.title( 'partitioned data', fontsize=14 )
# save plot
plt.savefig( PLOT_DIR + 'data0-partitioned.png' )
plt.show()
plt.close()
