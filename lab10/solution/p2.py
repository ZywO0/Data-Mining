#--
# p2.py
# letsios, sklar / 28 Jan 2021
# applies PCA to iris data set
# reads iris data from external file
#--

import sys
import csv
import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition

DEBUGGING = True
DATA_DIR  = '../data/'
DATA_FILE = 'iris.csv'



###### MAIN ######

#-get data from a file
try:
    df = pd.read_csv( DATA_DIR + DATA_FILE, na_filter=False )
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
#-get and size of raw data set
N = len( df.columns )
M = len( df.values )

#-print columns
if DEBUGGING:
    print('INPUT FILE = ' + DATA_DIR + DATA_FILE)
    print('number of attributes = ' + str( N ))
    print('number of instances = ' + str( M ))
    for ( i, c, t ) in zip( range( N ), df.columns, df.dtypes ):
        print('{} - {} ({})'.format( i, c, t ))

#-get target (class or label) and attribute values and list of feature names
# note that we have to convert class values (labels) to an integer (index)
y = np.zeros( M )
X = np.zeros(( M, N-1 ), dtype='float' )
feature_name_dict = { 'sepallength':0, 'sepalwidth':1, 'petallength':2, 'petalwidth':3 }
# define indexes for class values (labels)
class_dict = { 'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 }
for i in range( N ):
    if ( df.columns[i] == 'class' ):
        for j in range( M ):
            y[j] = class_dict[ df['class'][j] ]
    else:
        for j in range( M ):
            X[j][ feature_name_dict[ df.columns[i] ] ] = df[ df.columns[i] ][j]
N -= 1 # remove 'class' from number of attributes
if ( DEBUGGING ):
    print('shape of target=',np.shape( y ))
    print('shape of attributes=',np.shape( X ))
    for i in range( N ):
        print(list(feature_name_dict.keys())[i], np.mean( X[i] ))

#-get a list of the unique classes in the data set
the_classes = np.unique( y )
num_classes = len( the_classes )
if ( DEBUGGING ):
    print('unique list of classes=', the_classes)


#-for plotting, get a "cycle" (like a dictionary) of colors
colour_cycle = plt.rcParams['axes.prop_cycle']

#-generate a scatter plot matrix of pairwise feature comparisons
plotnum = 1
for row in range( N ):
    for col in range( N ):
        plt.subplot( N, N, plotnum )
        for myclass, mycolour in zip( range( num_classes ), colour_cycle ):
            data = X[y==myclass]
            plt.scatter( data[:,row], data[:,col], s=1, marker='.', c=mycolour['color'] )
            if ( col==0 ):
                plt.ylabel( list(feature_name_dict.keys())[row], fontsize=8 )
            if ( row==N-1 ):
                plt.xlabel( list(feature_name_dict.keys())[col], fontsize=8 )
            plt.tick_params( axis='both', labelsize=8 )
        plotnum += 1
plt.savefig( '../plots/iris2-scatter-matrix.png' )
plt.show()

#-initialise principal component analysis object
# here, we set the n_components argument to be equal to the number of
# features-1, so we can look at the impact of each possible reduction
# using PCA (i.e., to 1, 2, ..., N-1 features).  but if you
# know how many principal components you want, then you can adjust
# this argument accordingly. for example, if you only want the first
# principal component, then set n_components=1.
# see:
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
pca = decomposition.PCA( n_components=N-1 )

#-fit the PCA model using data set X
pca.fit( X )

#-apply dimensionality reduction to X
pca_X = pca.transform( X )
pca_N = pca_X.shape[1]

#-generate scatter plot matrix of pairwise feature comparisons using
#-the principal components just learned
plotnum = 1
for row in range( pca_N ):
    for col in range( pca_N ):
        plt.subplot( pca_N, pca_N, plotnum )
        for myclass, mycolour in zip( range( num_classes ), colour_cycle ):
            data = pca_X[y==myclass]
            plt.scatter( data[:,row], data[:,col], s=1, marker='.', c=mycolour['color'] )
            if ( col==0 ):
                plt.ylabel( 'PC-%d' % (row+1), fontsize=8 )
            if ( row==pca_N-1 ):
                plt.xlabel( 'PC-%d' % (col+1), fontsize=8 )
            plt.tick_params( axis='both', labelsize=8 )
        plotnum += 1
plt.savefig( '../plots/iris2-pca-scatter-matrix.png' )
plt.show()
