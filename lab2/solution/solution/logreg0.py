#--
# logreg0.py
# Builds a Logistic Regression Classifier using the Iris data set
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import numpy as np
import sklearn.datasets as data
import sklearn.model_selection as model_select
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocess
import matplotlib.pyplot as plt


STEP_SIZE = 0.1
FILENAME_TAG = '01'
DEBUGGING = False
PLOT_DIR = '../plots/'

# load iris data
iris = data.load_iris()
num_classes = len( iris.target_names )
if ( DEBUGGING ):
    print('classes = ', iris.target_names)
    print('attributes = ', iris.feature_names)

# we'll only look at two dimensions (0th and 1st)
# split the data into training and test sets
X = iris.data[:,0:2]
y = iris.target
X_train, X_test, y_train, y_test = model_select.train_test_split( X, y, random_state=0 )
M_train = len( X_train )
M_test = len( X_test )
if ( DEBUGGING ):
    print('number of training instances = ' + str( M_train ))
    print('number of test instances = ' + ( M_test ))

# initialise logistic regression model
clf = linear_model.LogisticRegression( solver='lbfgs', multi_class='multinomial' )

# fit model to iris data
clf.fit( X_train, y_train )


# find the decision boundary within a 2D grid space, where the two
# dimensions are comprised of the range of values for the two
# attributes we consider here. we use 2 attributes so we can visualise
# easily, but we could use more than two.

# set the limits of the 2D grid space, to make sure that all
# the points in the data set are fully visible in the plot
x0_min = np.min( X[:,0] ) - STEP_SIZE
x0_max = np.max( X[:,0] ) + STEP_SIZE
x1_min = np.min( X[:,1] ) - STEP_SIZE
x1_max = np.max( X[:,1] ) + STEP_SIZE
if ( DEBUGGING ):
    print(x0_min, x0_max, x1_min, x1_max)

# initialise two arrays, one for each attribute, that contains
# hypothetical values at evenly spaced intervals (STEP_SIZE),
# beginning with the minimum and going up to the maximum value for
# each (e.g., from x0_min to x0_max)
x0_range = np.arange( x0_min, x0_max, STEP_SIZE )
x0_len = len( x0_range ) # save number of items in the x0_range array
x1_range = np.arange( x1_min, x1_max, STEP_SIZE )
x1_len = len( x1_range ) # save number of items in the x1_range array
if ( DEBUGGING ):
    print('x0 range = ', x0_range)
    print('x0_range length = ', x0_len)
    print('x1 range = ', x1_range)
    print('x1_range length = ', x1_len)

# now we use our classifier to predict the class for all the possible
# combinations of values over the range of the two attributes we are
# using to test the classifier. in order to do this, we need to
# initialise an array that contains pairs of possible attribute
# values. these get sent to the "predict()" function of the
# classifier, which will return a parallel array of predicted classes
# for each attribute pair.
X_pairs = np.zeros(( x0_len * x1_len, 2 ))
if ( DEBUGGING ):
    print('X_pairs shape = ', np.shape( X_pairs ))
i = 0
for i1 in range( x1_len ):
    for i0 in range( x0_len ):
        X_pairs[i] = np.array( [ x0_range[i0], x1_range[i1] ] )
        i += 1
y_hat_pairs = clf.predict( X_pairs )



#-now let's compute some scores

# first, with the training set
print('training score = ', clf.score( X_train, y_train ))
print('test score = ', clf.score( X_test, y_test ))
print('mesh score = ', clf.score( X_pairs, y_hat_pairs ))
y_hat = clf.predict( X_test )

# compute accuracy score. note that we cannot compute this score for
# the y_hat_pairs data, because we do not have a "ground truth" like
# we do with the test set, i.e., "y_test".
print('accuracy score on test set = ', metrics.accuracy_score( y_test, y_hat ))

# compute the confusion matrix
cm = metrics.confusion_matrix( y_test, y_hat )
print('confusion matrix =')
# print '%10s\t%s' % ( ' ','predicted-->' )
print('\t predicted-->' )
# print '%10s\t' % ( 'actual:' ),
print( 'actual:', end='' )
for i in range( len( iris.target_names )):
    # print '%10s\t' % ( iris.target_names[i] ),
    print(iris.target_names[i], end='')
# print '\n',
print()
for i in range( len( iris.target_names )):
    # print '%10s\t' % ( iris.target_names[i] ),
    for j in range( len( iris.target_names )):
        # print '%10s\t' % ( cm[i,j] ),
        print( cm[i,j], end='' ),
    # print '\n',
    print()
# print '\n',
print()

# compute precision, recall and f1 scores
print('precision score = tp / (tp + fp) =')
precision = metrics.precision_score( y_test, y_hat, average=None )
for i in range( len( iris.target_names )):
    # print '\t%s = %f' % ( iris.target_names[i], precision[i] )
    print('\t {} = {}'.format( iris.target_names[i], precision[i] ))

print('recall score = tp / (tp + fn) =')
recall = metrics.recall_score( y_test, y_hat, average=None )
for i in range( len( iris.target_names )):
    # print '\t%s = %f' % ( iris.target_names[i], recall[i] )
    print('\t {} = {}'.format( iris.target_names[i], recall[i] ))

print('f1 score = 2 * (precision * recall) / (precision + recall) =')
f1 = metrics.f1_score( y_test, y_hat, average=None )
for i in range( len( iris.target_names )):
    # print '\t%s = %f' % ( iris.target_names[i], f1[i] )
    print('\t {} = {}'.format( iris.target_names[i], f1[i] ))

# we can compute a "decision function" which gives "confidence scores"
# corresponding to the samples. this is the signed distance from each
# value in X to the decision boundary. we can find the furthest from the boundary
conf_scores = clf.decision_function( X_pairs )
if ( DEBUGGING ):
    print('confidence scores shape =', np.shape( conf_scores ))
    print(conf_scores)
# compute ROC curve for each class
# binarize the output, since we have a multi-class data set
y_binary = preprocess.label_binarize( y_hat_pairs, classes=sorted( set( y )) )
if ( DEBUGGING ):
    print('y binary shape = ',np.shape( y_binary ))

# compute ROC curve
fpr = dict()
tpr = dict()
for c in range( num_classes ):
    (fpr[c], tpr[c], tmp) = metrics.roc_curve( y_binary[:,c], conf_scores[:,c] )

# plot ROC curve
plt.figure()
colours = [ ( 1, 0, 0, 1 ), ( 0, 1, 0, 0.7 ), ( 0, 0, 1, 0.5 ) ]
for c in range( num_classes ):
    plt.plot( fpr[c], tpr[c], c=colours[c], linewidth=2, label=iris.target_names[c] )
plt.legend( loc='lower right' )
min = -0.05
max = 1.05
plt.plot( [min, max], [min, max], color='k', linestyle='--' )
plt.xlim( [min, max] )
plt.ylim( [min, max] )
plt.xlabel( 'false positive (FP) rate' )
plt.ylabel( 'true positive (TP) rate' )
plt.savefig( PLOT_DIR + 'iris-log-reg-roc-'+FILENAME_TAG+'.png' )
plt.show()


# PLOT the decision boundary
plt.figure()
plt.set_cmap( 'Blues' )

# plot the training points
markers = [ 'o','<','s' ]
for i in range( M_train ):
    plt.plot( X_train[i,0], X_train[i,1], marker=markers[y_train[i]], markeredgecolor='w', markerfacecolor=colours[y_train[i]], markersize=9 )
# and also plot the test points
for i in range( M_test ):
    plt.plot( X_test[i,0], X_test[i,1], marker=markers[y_test[i]], markeredgecolor='w', markerfacecolor=colours[y_test[i]], markersize=9 )

# generate a 2D surface (mesh) that covers all possible combinations
# of the two attribute value ranges we just initialised above so that
# we can use it below to colour the decision boundaries.
x0_mesh, x1_mesh = np.meshgrid( x0_range, x1_range )
if ( DEBUGGING ):
    print('x0 mesh = ', x0_mesh)
    print('x0 mesh shape = ', np.shape( x0_mesh ))
    print('x1 mesh = ', x1_mesh)
    print('x1 mesh shape = ', np.shape( x1_mesh ))
y_hat_mesh = y_hat_pairs.reshape( x0_mesh.shape )
# x0_mesh and x1_mesh are the coordinates of the quadrilateral corners
# for each cell in the colour mesh and y_hat_mesh contains the
# corresponding class for that coordinate pair, which is used to set
# the colour for that quadrilateral.
# plt.pcolormesh( x0_mesh, x1_mesh, y_hat_mesh, shading='flat' )
plt.pcolormesh( x0_mesh, x1_mesh, y_hat_mesh, shading='auto')

# add labels to plot
plt.xlabel( iris.feature_names[0] )
plt.ylabel( iris.feature_names[1] )
plt.xlim( x0_min, x0_max )
plt.ylim( x1_min, x1_max )

# show it!
plt.savefig( PLOT_DIR + 'iris-log-reg-'+FILENAME_TAG+'.png' )
plt.show()
