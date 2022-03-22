#--
# p1.py
# classifying text as 'spam' or 'not spam'
# @author: letsios, sklar
# @created: 28 Jan 2021
#--

import sys
import csv
import random
import string
import pandas as pd
import numpy as np
import nltk
import sklearn.model_selection as modsel
import sklearn.naive_bayes as nb

DEBUGGING = True

DATA_DIR  = '../data/'
DATA_FILE = 'SMSSpamCollection.csv'



#--STEPS 1-2: OBTAIN CORPUS AND STATS

# read raw data set from csv file into data frame
try:
    df = pd.read_csv( DATA_DIR + DATA_FILE, na_filter=False, encoding='latin-1' )
except Exception as x:
    print('error> reading data file' + str( x ))
    sys.exit()

# get size of raw data set
M = len( df.values )
N = len( df.columns )
if ( DEBUGGING ):
    print('number of instances = ' + str( M ))
    print('number of columns = ' + str( N ))

# parse raw data set into a form ready for nltk and scikit-learn
msgs   = []
y_raw  = [] # labels
#numerr = 0  # tally of number of errors in the raw data set
for rec in df.values:
    try:
        label = rec[0].strip()
        msg   = rec[1].strip()
        # do this to make sure there aren't any unreadable chars in the raw data
        words = nltk.word_tokenize( msg )
        msgs.append( msg )
        y_raw.append( label )
    except Exception as x:
        print('error> parsing raw data file: ' + str( x ))
        #numerr += 1
        #print('warning> number of input errors = ' + str( numerr ))
if ( DEBUGGING ):
    print('msgs shape = ', np.shape( msgs ), 'y_raw shape = ', np.shape( y_raw ))

#-split data set into training and test sets
msgs_train, msgs_test, y_train, y_test = modsel.train_test_split( msgs, y_raw, test_size=0.50 )
if ( DEBUGGING ):
    print('size of data sets:')
    print('msgs:   training={} test={}'.format( len( msgs_train ), len( msgs_test )))
    print('labels: training={} test={}'.format( len( y_train ), len( y_test )))


#--STEP 3: CREATE BALANCED TRAINING SET

# split into 'ham' and 'spam' to make sure training set is balanced
ham  = []
spam = []
for ( msg, label ) in zip( msgs_train, y_train ):
    if ( label == 'ham' ):
        ham.append( msg )
    elif ( label == 'spam' ):
        spam.append( msg )
    else:
        numerr += 1 # count additional errors, if any
if ( DEBUGGING ):
    pass
    #print('number of: ham={} spam={} errors={} total={}'.format( len( ham ), len( spam ), numerr, ( len( ham ) + len( spam ) + numerr )))

# add spam, if necessary, by copying randomly chosen 'spam' messages
while( len( spam ) < len( ham )):
    i = random.randint( 0, len(spam)-1 )
    spam.append( spam[i] )

# add ham, if necessary, by copying randomly chosen 'ham' messages
while( len( ham ) < len( spam )):
    i = random.randint( 0, len(ham)-1 )
    ham.append( ham[i] )

if ( DEBUGGING ):
    print('balanced! number of: ham={} spam={}'.format( len(ham), len(spam) ))

msg_bal_train = []
y_bal_train = []
for j in range( len( ham )):
    msg_bal_train.append( ham[j] )
    y_bal_train.append( 'ham' )
for j in range( len( spam )):
    msg_bal_train.append( spam[j] )
    y_bal_train.append( 'spam' )

if ( DEBUGGING ):
    print('size of balanced training: msgs={} labels={}'.format( len( msg_bal_train ), len( y_bal_train )))


#--STEP 4: CHARACTERISE THE TRAINING SET

# attributes:
# num_words
# msg_len
# num_digits
# num_punct
# num_upper
num_attributes = 5
X = []
for msg in ( msg_bal_train ):
    # compute stats on message
    msg_len = len( msg )
    num_digits = 0
    num_punct = 0
    num_upper = 0
    for ch in msg:
        if ( ch.isdigit() ):
            num_digits += 1
        if ( ch in string.punctuation ):
            num_punct += 1
        if ( ch.isupper() ):
            num_upper += 1
    # split the message into words and compute stats on words
    words = nltk.word_tokenize( msg )
    num_words = len( words )
    # store attributes in matrix
    X.append(( num_words, msg_len, num_digits, num_punct, num_upper ))
X = np.array( X )
y_bal_train = np.array( y_bal_train )
if ( DEBUGGING ):
    print('X shape =', X.shape, 'y_bal_train shape =', y_bal_train.shape)


#--STEP 5: TRAIN CLASSIFIER

clf = nb.MultinomialNB()
clf.fit( X, y_bal_train )

# test classifier
num_TP = 0
num_TN = 0
num_FP = 0
num_FN = 0
num_err = 0
for ( msg, label ) in zip( msgs_test, y_test ):
    # compute stats on message
    msg_len = len( msg )
    num_digits = 0
    num_punct = 0
    num_upper = 0
    for ch in msg:
        if ( ch.isdigit() ):
            num_digits += 1
        if ( ch.isupper() ):
            num_upper += 1
        if ( ch in string.punctuation ):
            num_punct += 1
    # split the message into words and compute stats on words
    words = nltk.word_tokenize( msg )
    num_words = len( words )
    # store attributes in vector
    A = np.array(( num_words, msg_len, num_digits, num_punct, num_upper ))
    A = A.reshape( 1, -1 )
    # predict class
    pred_label = clf.predict( A ) #[ A ] )
    # tally result: spam = positive, ham = negative
    if ( label == 'spam' ):
        if ( pred_label == 'spam' ):
            num_TP += 1
        elif ( pred_label == 'ham' ):
            num_FN += 1
        else:
            num_err += 1
    elif ( label == 'ham' ):
        if ( pred_label == 'spam' ):
            num_FP += 1
        elif ( pred_label == 'ham' ):
            num_TN += 1
        else:
            num_err += 1
    else:
        num_err += 1

# any data errors?
if ( num_err > 0 ):
    print('number of errors in data set = ' + str( num_err ))

# print confusion matrix
print('TP={} FP={} TN={} FN={}'.format( num_TP, num_FP, num_TN, num_FN ))

precision = num_TP / float( num_TP + num_FP )
recall = num_TP / float( num_TP + num_FN )
f1 = 2 * precision * recall / ( precision + recall )
print('precision={} recall={} f1={}'.format( precision, recall, f1 ))
