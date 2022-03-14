#--
# ib1h.py
# instance-based model, where all attributes are encoded as numeric.
# model is used to find nearest neighbour to MY_HOUSE (defined below).
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

MY_HOUSE = { 'HomeID' : -999, 'Price' : -999, 'SqFt' : 2050, 'Bedrooms' : 2, 'Bathrooms' : 1, 'Offers' : 2, 'Brick' : 'No', 'Neighborhood' : 'East' }
K = 5


#--
# dist_square()
# This function returns the squared difference between two scalar
# arguments, v0 and v1.
#--
def dist_square( v0, v1 ):
    d = math.pow( v0 - v1, 2 )
    return d

#--
# dist_euclidean()
# This function returns the Euclidean distance between two instances,
# X0 and X1, each of which is an array of scalar attribute values.
# Note that this ignores the first two values in the attribute arrays
# (HomeID and Price).
#--
def dist_instance_euclidean( X0, X1, N ):
    d = 0.0
    for i in range( 2, N ):
        d += dist_square( X0[i], X1[i] ) 
    d = math.sqrt( d )
    return d

#--
# dist_manhattan()
# This function returns the Manhattan distance between two instances,
# X0 and X1, each of which is an array of scalar attribute values.
# Note that this function ignores the first two values in the
# attribute arrays (HomeID and Price).
#--
def dist_instance_manhattan( X0, X1, N ):
    d = 0.0
    for i in range( 2, N ):
        d += abs( X0[i] - X1[i] ) 
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
# 0 - HomeID (int64)
# 1 - Price (int64)
# 2 - SqFt (int64)
# 3 - Bedrooms (int64)
# 4 - Bathrooms (int64)
# 5 - Offers (int64)
# 6 - Brick (object)
# 7 - Neighborhood (object)
    print('possible values of Brick = ', set( df['Brick'] ))
    print('possible values of Neighborhood = ', set( df['Neighborhood'] ))

#-encode all variables as numeric
X = np.zeros(( M, N ))
print(np.shape( X ))
for j in range( M ):
    for i in range( N ):
        if ( df.columns[i] == 'Brick' ):
            X[j][i] = ENCODE_BRICK[ df.values[j][i] ]
        elif ( df.columns[i] == 'Neighborhood' ):
            X[j][i] = ENCODE_HOOD[ df.values[j][i] ]
        else:
            X[j][i] = df.values[j][i]

#-encode MY_HOUSE instance
home = np.zeros( N )
for i in range( N ):
    attribute = df.columns[i]
    if ( df.columns[i] == 'Brick' ):
        home[i] = ENCODE_BRICK[ MY_HOUSE[ attribute ]]
    elif ( df.columns[i] == 'Neighborhood' ):
        home[i] = ENCODE_HOOD[ MY_HOUSE[ attribute ]]
    else:
        home[i] = MY_HOUSE[attribute]
if DEBUGGING:
    print('home=', home)

#-compute distance from "home" to instances
dist_euclidean = np.zeros( M )
dist_manhattan = np.zeros( M )
for j in range( M ):
    dist_euclidean[j] = dist_instance_euclidean( home, X[j], N )
    dist_manhattan[j] = dist_instance_manhattan( home, X[j], N )

#-report K nearest neighbours
# euclidean
sorted_dist = np.argsort( dist_euclidean )
print('{} closest according to Euclidean distance:'.format( K ))
prices = 0.0
for i in range( K ):
    print('({}) HomeID={} distance={}'.format( i, X[sorted_dist[i]][0], dist_euclidean[sorted_dist[i]] ))
    prices += df['Price'][sorted_dist[i]]
prices /= K
print('average price = $' + str( prices ))
# manhattan
sorted_dist = np.argsort( dist_manhattan )
prices = 0.0
print('{} closest according to Manhattan distance:'.format( K ))
for i in range( K ):
    print('({}) HomeID={} distance={}'.format( i, X[sorted_dist[i]][0], dist_manhattan[sorted_dist[i]] ))
    prices += df['Price'][sorted_dist[i]]
prices /= K
print('average price = $' + str( prices ))
