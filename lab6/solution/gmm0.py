#--
# gmm0.py
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
import matplotlib.pyplot as plt

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
X = np.zeros(( M, 2 ))
print(np.shape( X ))
for j in range( M ):
    X[j][0] = df['Price'][j]
    X[j][1] = df['SqFt'][j]

#-fit gaussian mixture model
K = 2
gmm = mixture.GaussianMixture( n_components=K, covariance_type='spherical' )
gmm.fit( X )

#-predict labels (components) for this data
labels = gmm.predict( X )

#-print out the means for each component
print('GMM parameters:')
for k in range( K ):
    print('component {}:'.format( k ))
    for i in range( 2 ):
        print('attribute {}, mean={}'.format( i, gmm.means_[k][i] ))
    print()

#-compute and print log-likelihood score
print('log-likelihood score=' + str( gmm.score( X ) ))

#-plot
# raw data
plt.figure()
plt.scatter( X[:,0], X[:,1], 0.8 )
plt.xlabel( 'Price' )
plt.ylabel( 'SqFt' )
plt.savefig( PLOTS_DIR + 'gmm0-raw-data.png' )
plt.show()
plt.close()
# clusters as determined by GMM
plt.figure()
plt.scatter( X[:,0], X[:,1], 0.8 )
plt.xlabel( 'Price' )
plt.ylabel( 'SqFt' )
plt.plot( gmm.means_[0][0], gmm.means_[0][1], 'gs', markersize=10 )
plt.plot( gmm.means_[1][0], gmm.means_[1][1], 'mo', markersize=10 )
for j in range( len( labels )):
    if ( labels[j] == 0 ):
        plt.plot( X[j,0], X[j,1], 'gs', alpha=0.5, markersize=5 )
    else:
        plt.plot( X[j,0], X[j,1], 'mo', alpha=0.5, markersize=5 )
plt.savefig( PLOTS_DIR + 'gmm0-clusters.png' )
plt.show()
plt.close()
