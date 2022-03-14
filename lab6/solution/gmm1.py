#--
# gmm1.py
# gaussian mixture model
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import sys
import csv
import pandas as pd
import numpy as np
import sklearn.mixture as mixture
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

DEBUGGING = False
DATA_DIR  = '../data/'
DATA_FILE = 'house-prices.csv'
PLOTS_DIR = '../plots/'

ENCODE_BRICK = { 'No' : 0 , 'Yes' : 1 }
ENCODE_HOOD  = { 'East' : 0, 'North' : 90, 'West' : 180, 'South' : 270 }

# define markers for up to 10 clusters
CLUSTER_MARKERS = [ 'bo', 'rv', 'c^', 'm<', 'y>', 'ks', 'bp', 'r*', 'cD', 'mP' ]



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
X = np.zeros(( M, 2 ))
print(np.shape( X ))
for j in range( M ):
    X[j][0] = df['Price'][j]
    X[j][1] = df['SqFt'][j]

#-plot raw data
plt.figure()
plt.scatter( X[:,0], X[:,1], 0.8 )
plt.xlabel( 'Price' )
plt.ylabel( 'SqFt' )
plt.savefig( PLOTS_DIR + 'gmm1-raw-data.png' )
plt.show()
plt.close()

#-loop through different values for K (number of components)
for K in range( 2, 10 ):

    #-fit gaussian mixture model
    gmm = mixture.GaussianMixture( n_components=K, covariance_type='spherical' )
    gmm.fit( X )

    #-predict labels (components) for this data
    labels = gmm.predict( X )

    #-print out the means for each component
    if DEBUGGING:
        print('GMM parameters:')
        for k in range( K ):
            print('component {}:'.format( k ))
            for i in range( 2 ):
                print('attribute {}, mean={}'.format( i, gmm.means_[k][i] ))
            print()

    #-compute log-likelihood score (minimise)
    ll_score = ( gmm.score( X ) )

    #-compute silhouette score (maximise)
    sc_score = metrics.silhouette_score( X, labels, metric='euclidean' )

    #-compute calinski-harabaz score: ratio between the within-cluster and between-cluster (maximise)
    ch_score = metrics.calinski_harabasz_score( X, labels )

    #-print results for this value of K
    print('K={}  log-likelihood={}  silhouette={}  calinski-harabasz={}'.format( K, ll_score, sc_score, ch_score ))

    #-plot components (clusters) as determined by GMM
    plt.figure()
    plt.scatter( X[:,0], X[:,1], 0.8 )
    plt.xlabel( 'Price' )
    plt.ylabel( 'SqFt' )
    for k in range( K ):
        mymarker = CLUSTER_MARKERS[ k ]
        plt.plot( gmm.means_[k][0], gmm.means_[k][1], mymarker, markersize=10 )
    for j in range( len( labels )):
        mymarker = CLUSTER_MARKERS[ labels[j] ]
        plt.plot( X[j,0], X[j,1], mymarker, alpha=0.5, markersize=5 )
    plt.title( 'K=' + str( K ))
    plt.savefig( PLOTS_DIR + 'gmm1-K' + str( K ) + '.png' )
    plt.show()
    plt.close()
