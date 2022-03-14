#--
# ib0h.py
# instance-based model, where attributes are measured different if they are numeric vs nominal
# @author: letsios, sklar
# @created: 12 Jan 2021
#--

import sys
import csv
import pandas as pd
import numpy as np
import math

DEBUGGING = True
DATA_DIR  = '../data/'
DATA_FILE = 'house-prices.csv'

ENCODE_BRICK = { 'No' : 0 , 'Yes' : 1 }
ENCODE_HOOD  = { 'East' : 0, 'North' : 90, 'West' : 180, 'South' : 270 }

MY_HOUSE = { 'HomeID' : [ -999 ], 'Price' : [ -999 ], 'SqFt' : [ 2050 ], 'Bedrooms' : [ 2 ], 'Bathrooms' : [ 1 ], 'Offers' : [ 2 ], 'Brick' : [ 'No' ], 'Neighborhood' : [ 'East' ] }
K = 5


#--
# dist_nominal()
# This function returns the distance between two nominal arguments, v0
# and v1. The function returns 0 if the arguments are the same and 1
# otherwise.
#--
def dist_nominal( v0, v1 ):
    if ( v0 == v1 ):
        d = 0.0
    else:
        d = 1.0
    return d


#--
# dist_square()
# This function returns the squared difference between two scalar
# arguments, v0 and v1.
#--
def dist_square( v0, v1 ):
    d = math.pow( v0 - v1, 2 )
    return d


#--
# dist_instance_euclidean
# This function computes the distance between two instances, rows X0
# and X1 in data frames df0 and df1, respectively. The function takes
# into account that some attributes are nominal (categorical) and some
# are numeric.  The Euclidean distance metric is used to compute the
# distance across all the attributes in the instances.
# Note that this function ignores the first two attributes (HomeID and Price).
#--
def dist_instance_euclidean( df0, X0, df1, X1 ):
    N = len( df0.columns )
    d = 0.0
    for i in range( 2, N ):
        if ( df0.dtypes[i] == 'object' ):
            d += dist_nominal( df0[df0.columns[i]][X0], df1[df0.columns[i]][X1] )
        else:
            d += dist_square( df0[df0.columns[i]][X0], df1[df0.columns[i]][X1] )
    d = math.sqrt( d )
    return d


#--
# dist_instance_manhattan
# This function computes the distance between two instances, rows X0
# and X1 in data frames df0 and df1, respectively. The function takes
# into account that some attributes are nominal (categorical) and some
# are numeric.  The Manhattan distance metric is used to compute the
# distance across all the attributes in the instances.
# Note that this function ignores the first two attributes (HomeID and Price).
#--
def dist_instance_manhattan( df0, X0, df1, X1 ):
    N = len( df0.columns )
    d = 0.0
    for i in range( 2, N ):
        if ( df0.dtypes[i] == 'object' ):
            d += dist_nominal( df0[df0.columns[i]][X0], df1[df0.columns[i]][X1] )
        else:
            d += abs( df0[df0.columns[i]][X0] - df1[df0.columns[i]][X1] )
    return d


#--
# MAIN
#--

#-get data from a file
try:
    df = pd.read_csv( DATA_DIR + DATA_FILE, na_filter=False )
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()

#-get size of raw data set
N = len( df.columns )
M = len( df.values )

#-print columns
if DEBUGGING:
    print('INPUT FILE = ' + DATA_DIR + DATA_FILE)
    print('number of attributes = ' + str( N ))
    print('number of instances = ' + str( M ))
    for ( i, c, t ) in zip( range( N ), df.columns, df.dtypes ):
        print('{} - {} ({})'.format( i, c, t ))
    print('possible values of Brick = ', set( df['Brick'] ))
    print('possible values of Neighborhood = ', set( df['Neighborhood'] ))

#-encode MY_HOUSE instance in a data frame
home = pd.DataFrame.from_dict( MY_HOUSE )
print(home)

#-compute distance from "home" to instances
dist_euclidean = np.zeros( M )
dist_manhattan = np.zeros( M )
for j in range( M ):
    dist_euclidean[j] = dist_instance_euclidean( df, j, home, 0 )
    dist_manhattan[j] = dist_instance_manhattan( df, j, home, 0 )

#-report K nearest neighbours
# euclidean
sorted_dist = np.argsort( dist_euclidean )
print('{} closest according to Euclidean distance:'.format( K ))
prices = 0.0
for i in range( K ):
    print('({}) HomeID={} distance={}'.format( i, df['HomeID'][sorted_dist[i]], dist_euclidean[sorted_dist[i]] ))
    prices += df['Price'][sorted_dist[i]]
prices /= K
print('average price = $' + str( prices ))
# manhattan
sorted_dist = np.argsort( dist_manhattan )
prices = 0.0
print('{} closest according to Manhattan distance:'.format( K ))
for i in range( K ):
    print('({}) HomeID={} distance={}'.format( i, df['HomeID'][sorted_dist[i]], dist_manhattan[sorted_dist[i]] ))
    prices += df['Price'][sorted_dist[i]]
prices /= K
print('average price = $' + str(prices ))
