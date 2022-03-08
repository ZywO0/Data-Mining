#--
# data0.py
# Loads Iris data set and computes information about it.
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import numpy as np
import sklearn.datasets as data

iris = data.load_iris()
print('classes = ', iris.target_names)
print('attributes = ', iris.feature_names)
# iris.data = X values
# iris.target = y values
X = iris.data
M = len( iris.data )
print('number of instances = ' + str(M) )

x0_min = np.min( X[:,0] )
x0_max = np.max( X[:,0] )
x1_min = np.min( X[:,1] )
x1_max = np.max( X[:,1] )
print(iris.feature_names[0], x0_min, x0_max)
print(iris.feature_names[1], x1_min, x1_max)

