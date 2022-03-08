#--
# oner0.py
# Builds a set of classification rules using oneR on the Iris data set
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import random
import numpy as np
import sklearn.datasets as data

DEBUGGING = True


# load the built-in iris data set
iris = data.load_iris()
X = iris.data
y = iris.target
M = len( X ) # number of instances
N = len( iris.feature_names ) # number of attributes
if ( DEBUGGING ):
    print('classes = ', iris.target_names)
    print('attributes = ', iris.feature_names)
    print('number of instances = ' + str( M ))
    for i in range( M ):
        print(X[i], y[i])

# get unique list of class values
classes = sorted( set( y ))
num_classes = len( classes )
if ( DEBUGGING ):
    print('classes = ', classes)

# get unique lists of values for each attribute
attr_values = []
for i in range( N ):
    attr_values.append( sorted( set( X[:,i] )))
if ( DEBUGGING ):
    print('attribute values =')
    for i in range( N ):
        print(iris.feature_names[i], attr_values[i])

attr_dict = [[ dict() for k in range( len( attr_values[i] )) ] for i in range( N )]
for i in range( N ):
    for k in range( len( attr_values[i] )):
        for c in range( num_classes ):
            attr_dict[i][k][c] = 0

# for each attribute value, count the number of occurrences of each class
for j in range( M ): # loop through all instances
    c = y[j] # save the class for this instance
    for i in range( N ):
        # find index of attribute value X[j,i] in attributes[i]
        k = attr_values[i].index( X[j][i] )
        if ( k < 0 ):
            print('ERROR! (attribute, value) not found: (', i, X[j][i], ')')
        else:
            attr_dict[i][k][c] += 1

# for each attribute value, find the most frequent class (0, 1 or 2)
attr_value_class = [[ -1 for k in range( len( attr_values[i] )) ] for i in range( N ) ]
for i in range( N ):
    for k in range( len( attr_values[i] )):
        most_freq = 0
        for c in range( num_classes ):
            if ( attr_dict[i][k][c] > attr_dict[i][k][most_freq] ):
                most_freq = c
        attr_value_class[i][k] = most_freq

# display table of counts and most frequent class for each single attribute
if ( DEBUGGING ):
    for i in range( N ):
        for k in range( len( attr_values[i] )):
            print('attribute ', iris.feature_names[i])
            print(' value ', attr_values[i][k])
            print(' count = (' )
            for c in range( num_classes ):
                print(attr_dict[i][k][c])
            print(')  most frequent=', attr_value_class[i][k])

# build 1R set of rules using attributes 0 and 1
rules = [[ -1 for i0 in range( len( attr_values[0] )) ] for i1 in range( len( attr_values[1] )) ]
for k1 in range( len( attr_values[1] )):
    for k0 in range( len( attr_values[0] )):
        most_freq_0 = attr_value_class[0][k0]
        most_freq_1 = attr_value_class[1][k1]
        if ( most_freq_0 == most_freq_1 ):
            # the same class is the most frequent for both attribute values
            rule_class = most_freq_0
        elif ( attr_dict[0][k0][most_freq_0] > attr_dict[1][k1][most_freq_1] ):
            # the k0-th attribute value occurs more frequently than
            # the k1-th attribute value, so select the k0-th class
            rule_class = most_freq_0
        elif ( attr_dict[1][k1][most_freq_1] > attr_dict[0][k0][most_freq_0] ):
            # the k1-th attribute value occurs more frequently than
            # the k0-th attribute value, so select the k1-th class
            rule_class = most_freq_1
        else:
            # the k0-th and k1-th attributes occur with the same
            # frequency, to randomly select between them to decide
            # which class
            if ( random.random() < 0.5 ):
                rule_class = most_freq_0
            else:
                rule_class = most_freq_1
        rules[k1][k0] = rule_class

# display the rules
print('\nAnd the RULES are...')
if ( DEBUGGING ):
    for k1 in range( len( attr_values[1] )):
        for k0 in range( len( attr_values[0] )):
            print('({},{})=({},{}) -> {} ({})'.format( iris.feature_names[1], iris.feature_names[0], attr_values[1][k1], attr_values[0][k0], rules[k1][k0], iris.target_names[rules[k1][k0]] )) 

# score the rules
count = 0.0
# loop through instances
for j in range( M ):
    # find index in attribute values lists for the attribute values in this instance
    k0 = attr_values[0].index( X[j,0] )
    if ( k0 < 0 ):
        print('ERROR finding k0-th attribute: ', X[j,0])
        sys.exit()
    k1 = attr_values[1].index( X[j,1] )
    if ( k1 < 0 ):
        print('ERROR finding k1-th attribute: ', X[j,1])
        sys.exit()
    if ( y[j] == rules[k1][k0] ):
        count += 1
score = ( count / M )
print('number of correct predictions = {} out of {} = {}'.format( count, M, score ))
