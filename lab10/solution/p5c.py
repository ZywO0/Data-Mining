#--
# p5c.py
# letsios, sklar / 28 Jan 2021
# applies PCA to weather data set and builds a decision tree
#--

import sys
import csv
import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.model_selection as model_select
import sklearn.tree as tree
import sklearn.metrics as metrics

DEBUGGING = True
DATA_DIR  = '../data/'
DATA_FILE = 'weather.csv'



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
        print('%2d - %s (%s)' % ( i, c, t ))

#-get target (class or label) and attribute values and list of feature names
# note that we have to convert class values (labels) to an integer (index)
y = np.zeros( M )
X = np.zeros(( M, N-1 ), dtype='float' )
feature_name_dict = { 'outlook':0, 'temperature':1, 'humidity':2, 'windy':3 }
xi = 0
# define indexes for nominal attribute values
outlook_dict = { 'sunny':0, 'overcast':1, 'rainy':2 }
# define indexes for class values (labels)
class_dict = { 'no':0, 'yes':1 }
for i in range( N ):
    print(i, df.columns[i])
    if ( df.columns[i] == 'class' ):
        for j in range( M ):
            y[j] = class_dict[ df['class'][j] ]
    elif ( df.columns[i] == 'outlook' ):
        for j in range( M ):
            X[j][ feature_name_dict[ df.columns[i] ] ] = outlook_dict[ df[ df.columns[i] ][j] ]
    elif ( df.columns[i] == 'windy' ):
        for j in range( M ):
            if ( df[ df.columns[i] ][j] ):
                X[j][ feature_name_dict[ df.columns[i] ] ] = 1
            else:
                X[j][ feature_name_dict[ df.columns[i] ] ] = 0
    else:
        for j in range( M ):
            X[j][ feature_name_dict[ df.columns[i] ] ] = df[ df.columns[i] ][j]
N -= 1 # remove 'class' from number of attributes
if ( DEBUGGING ):
    print('shape of target=', np.shape( y ))
    print('shape of attributes=', np.shape( X ))
    for i in range( N ):
        print(list(feature_name_dict.keys())[i], np.mean( X[i] ))

#-get a list of the unique classes in the data set
the_classes = np.unique( y )
num_classes = len( the_classes )
if ( DEBUGGING ):
    print('unique list of classes=', the_classes)

#-initialise principal component analysis object
pca = decomposition.PCA( n_components=N-1 )

#-fit the PCA model using data set X
pca.fit( X )

#-apply dimensionality reduction to X
pca_X = pca.transform( X )
pca_N = pca_X.shape[1]


#-initialise a decision tree classifier
print('DECISION TREE ON RAW DATA:')
clf = tree.DecisionTreeClassifier( random_state = 0 )

# split the raw data into training and test sets
X_train, X_test, y_train, y_test = model_select.train_test_split( X, y, random_state=0 )
M_train = len( X_train )
M_test = len( X_test )
if ( DEBUGGING ):
    print('number of training instances = ' + str( M_train ))
    print('number of test instances = ' + str( M_test ))

# fit the tree model to the training data
clf.fit( X_train, y_train )

# predict the labels for the training and test sets
y_hat = clf.predict( X_train )
print('training accuracy = ', metrics.accuracy_score( y_train, y_hat ))
y_hat = clf.predict( X_test )
print('training accuracy = ', metrics.accuracy_score( y_test, y_hat ))

#-initialise a decision tree classifier
print('DECISION TREE ON PCA DATA:')
clf = tree.DecisionTreeClassifier( random_state = 0 )

# split the PCA data into training and test sets
X_train, X_test, y_train, y_test = model_select.train_test_split( pca_X, y, random_state=0 )
M_train = len( X_train )
M_test = len( X_test )
if ( DEBUGGING ):
    print('number of training instances = %d' + str( M_train ))
    print('number of test instances = ' + str( M_test ))

# fit the tree model to the training data
clf.fit( X_train, y_train )

# predict the labels for the training and test sets
y_hat = clf.predict( X_train )
print('training accuracy = ', metrics.accuracy_score( y_train, y_hat ))
y_hat = clf.predict( X_test )
print('training accuracy = ', metrics.accuracy_score( y_test, y_hat ))
