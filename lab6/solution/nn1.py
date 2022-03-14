#--
# nn1.py
# instance-based...  nearest neighbors
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import sys
import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.neighbors as neighbors

DEBUGGING = True
DATA_DIR  = '../data/'
DATA_FILE = 'house-prices.csv'

ENCODE_BRICK = { 'No' : 0 , 'Yes' : 1 }
ENCODE_HOOD  = { 'East' : 0, 'North' : 90, 'West' : 180, 'South' : 270 }

MY_HOUSE = { 'Price' : 0, 'SqFt' : 2050, 'Bedrooms' : 2, 'Bathrooms' : 1, 'Offers' : 2, 'Brick' : 'No', 'Neighborhood' : 'East' }
K = 5


#--
# MAIN
#--

#-get data from a file
try:
    df = pd.read_csv( DATA_DIR + DATA_FILE, na_filter=False )
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
#-get and size of raw data set
N = len( df.columns )
M = len( df.values ) #10 

#-print columns
if DEBUGGING:
    print('INPUT FILE = ' + DATA_DIR + DATA_FILE)
    print('number of attributes = ' + str( N ))
    print('number of instances = ' + str( M ))
    for ( i, c, t ) in zip( range( N ), df.columns, df.dtypes ):
        print('{} - {} ({})'.format( i, c, t ))
    print('possible values of Brick = ', set( df['Brick'] ))
    print('possible values of Neighborhood = ', set( df['Neighborhood'] ))

#-encode all variables as numeric
# do not encode the first two columns (HomeID and Price)
X = np.zeros(( M, N-2 ))
print(np.shape( X ))
for j in range( M ):
    for i in range( 2, N ):
        if ( df.columns[i] == 'Brick' ):
            X[j][i-2] = ENCODE_BRICK[ df.values[j][i] ]
        elif ( df.columns[i] == 'Neighborhood' ):
            X[j][i-2] = ENCODE_HOOD[ df.values[j][i] ]
        else:
            X[j][i-2] = df.values[j][i]

#-encode MY_HOUSE instance
home = np.zeros( N-2 )
for i in range( 2, N ):
    attribute = df.columns[i]
    if ( df.columns[i] == 'Brick' ):
        home[i-2] = ENCODE_BRICK[ MY_HOUSE[ attribute ]]
    elif ( df.columns[i] == 'Neighborhood' ):
        home[i-2] = ENCODE_HOOD[ MY_HOUSE[ attribute ]]
    else:
        home[i-2] = MY_HOUSE[attribute]
if DEBUGGING:
    print('home=', home)
home = home.reshape( 1, -1 )
print(np.shape( home ))
print(np.shape( X ))

#-compute K nearest neighbours
nn_euclidean = neighbors.NearestNeighbors( n_neighbors=K, algorithm='kd_tree', metric='euclidean' )
nn_manhattan = neighbors.NearestNeighbors( n_neighbors=K, algorithm='kd_tree', metric='manhattan' )
nn_euclidean.fit( X )
nn_manhattan.fit( X )
dist_euc, ind_euc = nn_euclidean.kneighbors( home, return_distance=True )
dist_man, ind_man = nn_manhattan.kneighbors( home, return_distance=True )
print(np.shape( dist_euc ))
print(np.shape( ind_euc ))


#-report nearest neighbours
# euclidean
print('{} closest according to Euclidean distance:'.format( K ))
prices = 0.0
for i in range( K ):
    print('({}) HomeID={} distance={}'.format( i, df['HomeID'][ind_euc[0][i]], dist_euc[0][i] ))
    prices += df['Price'][ind_euc[0][i]]
prices /= K
print('average price = $' + str( prices ))
# euclidean
print('{} closest according to Manhattan distance:'.format( K ))
prices = 0.0
for i in range( K ):
    print('({}) HomeID={} distance={}'.format( i, df['HomeID'][ind_man[0][i]], dist_man[0][i] ))
    prices += df['Price'][ind_euc[0][i]]
prices /= K
print('average price = $' + str( prices ))
