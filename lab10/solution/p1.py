#--
# p1.py
# letsios, sklar/28 Jan 2021
# applies PCA to iris data set
# uses iris data from sklearn datasets
#--

import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import sklearn.decomposition as decomposition


DEBUGGING = True


###### MAIN ######

#-load iris data set
iris = datasets.load_iris()
X = iris.data
y = iris.target
M = X.shape[0] # number of instances in the data set
N = X.shape[1] # number of features in the data set
if ( DEBUGGING ):
    print('shape of target=',np.shape( y ))
    print('shape of attributes=',np.shape( X ))
    print('mean of attributes=')
    for i in range( N ):
        print(iris.feature_names[i], np.mean( X[i] ))

#-get a list of the unique classes in the data set
the_classes = np.unique( y )
num_classes = len( the_classes )
if ( DEBUGGING ):
    print('unique list of classes=', the_classes)

#-for plotting, get a "property cycle" (like a dictionary) of colors
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
                plt.ylabel( iris.feature_names[row], fontsize=8 )
            if ( row==N-1 ):
                plt.xlabel( iris.feature_names[col], fontsize=8 )
            plt.tick_params( axis='both', labelsize=8 )
        plotnum += 1
plt.savefig( '../plots/iris-scatter-matrix.png' )
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
plt.savefig( '../plots/iris-pca-scatter-matrix.png' )
plt.show()
