#--
# nn0.py
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
PLOTS_DIR = '../plots/'

ENCODE_BRICK = { 'No' : 0 , 'Yes' : 1 }
ENCODE_HOOD  = { 'East' : 0, 'North' : 90, 'West' : 180, 'South' : 270 }


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
# do not encode the first column (HomeID)
X = np.zeros(( M, N-1 ))
print(np.shape( X ))
for j in range( M ):
    for i in range( 1, N ):
        if ( df.columns[i] == 'Brick' ):
            X[j][i-1] = ENCODE_BRICK[ df.values[j][i] ]
        elif ( df.columns[i] == 'Neighborhood' ):
            X[j][i-1] = ENCODE_HOOD[ df.values[j][i] ]
        else:
            X[j][i-1] = df.values[j][i]

#-compute K nearest neighbours
K = M
nn_euclidean = neighbors.NearestNeighbors( n_neighbors=K, algorithm='kd_tree', metric='euclidean' )
nn_manhattan = neighbors.NearestNeighbors( n_neighbors=K, algorithm='kd_tree', metric='manhattan' )
nn_euclidean.fit( X )
nn_manhattan.fit( X )
dist_euc, ind_euc = nn_euclidean.kneighbors( X, return_distance=True )
dist_man, ind_man = nn_manhattan.kneighbors( X, return_distance=True )
print(np.shape( dist_euc ))
print(np.shape( ind_euc ))

#-tally distances between instances
dist_mesh_euclidean = np.zeros(( M, M ))
dist_mesh_manhattan = np.zeros(( M, M ))
for j in range( M ):
    for i in range( M ):
        dist_mesh_euclidean[j][ ind_euc[j][i] ] = dist_euc[j][i]
        dist_mesh_manhattan[j][ ind_man[j][i] ] = dist_man[j][i]
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
plt.savefig( PLOTS_DIR + 'nn0-mesh-euclidean.png' )
plt.show()
plt.close()
# manhattan
plt.figure()
plt.set_cmap( 'Blues' )
plt.pcolormesh( x0_mesh, x1_mesh, dist_mesh_manhattan, shading='auto' )
plt.title( 'Manhattan distances' )
plt.savefig( PLOTS_DIR + 'nn0-mesh-manhattan.png' )
plt.show()
plt.close()
