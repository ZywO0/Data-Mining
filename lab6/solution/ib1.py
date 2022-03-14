#--
# ib1.py
# instance-based model, where all attributes are encoded as numeric
# @author: letsios, sklar
# @created: 12 Jan 2021
#--

import sys
import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


DEBUGGING = True
DATA_DIR  = '../data/'
DATA_FILE = 'house-prices.csv'
PLOTS_DIR = '../plots/'

ENCODE_BRICK = { 'No' : 0 , 'Yes' : 1 }
ENCODE_HOOD  = { 'East' : 0, 'North' : 90, 'West' : 180, 'South' : 270 }


#--
# dist_square()
# This function returns the squared difference between two scalar
# arguments, v0 and v1.
#--
def dist_square( v0, v1 ):
    d = math.pow( v0 - v1, 2 )
    return d


#--
# dist_instance_euclidean()
# This function returns the Euclidean distance between two instances,
# X0 and X1, each of which is an array of scalar attribute values.
# Note that the first attribute (HomeID) is ignored.
#--
def dist_instance_euclidean( X0, X1, N ):
    d = 0.0
    for i in range( 1, N ):
        d += dist_square( X0[i], X1[i] ) 
    d = math.sqrt( d )
    return d


#--
# dist_instance_manhattan()
# This function returns the Manhattan distance between two instances,
# X0 and X1, each of which is an array of scalar attribute values.
# Note that the first attribute (HomeID) is ignored.
#--
def dist_instance_manhattan( X0, X1, N ):
    d = 0.0
    for i in range( 1, N ):
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

#-compute distances between instances
dist_mesh_euclidean = np.zeros(( M, M ))
dist_mesh_manhattan = np.zeros(( M, M ))
for j0 in range( M ):
    for j1 in range( j0+1, M ):
        dist = dist_instance_euclidean( X[j0], X[j1], N )
        dist_mesh_euclidean[j0][j1] = dist
        dist_mesh_euclidean[j1][j0] = dist
        dist = dist_instance_manhattan( X[j0], X[j1], N )
        dist_mesh_manhattan[j0][j1] = dist
        dist_mesh_manhattan[j1][j0] = dist
print('Euclidean: mean distance = {} ({}), minimum distance = {}, maximum distance = {}'.format( np.mean( dist_mesh_euclidean ), np.std( dist_mesh_euclidean ), np.min( dist_mesh_euclidean ), np.max( dist_mesh_euclidean )))
print('Manhattan: mean distance = {} ({}), minimum distance = {}, maximum distance = {}'.format( np.mean( dist_mesh_manhattan ), np.std( dist_mesh_manhattan ), np.min( dist_mesh_manhattan ), np.max( dist_mesh_manhattan )))

#-make heatmap of instance differences
x0_range = np.arange( M )
x1_range = np.arange( M )
x0_mesh, x1_mesh = np.meshgrid( x0_range, x1_range )

#-plot the heatmaps, one for euclidean and one for manhattan
# euclidean
plt.figure()
plt.set_cmap( 'Blues' )
plt.pcolormesh( x0_mesh, x1_mesh, dist_mesh_euclidean, shading='auto' )
plt.title( 'Euclidean distances' )
plt.savefig( PLOTS_DIR + 'ib1-mesh-euclidean.png' )
plt.show()
plt.close()
# manhattan
plt.figure()
plt.set_cmap( 'Blues' )
plt.pcolormesh( x0_mesh, x1_mesh, dist_mesh_manhattan, shading='auto' )
plt.title( 'Manhattan distances' )
plt.savefig( PLOTS_DIR + 'ib1-mesh-manhattan.png' )
plt.show()
plt.close()
