#--
# p5.py
# # finds two closest documents using frequency term-document matrix and cosine similarity
# @author: letsios, sklar
# @created: 28 Jan 2021
#--

import textblob
import requests
import numpy as np

DEBUGGING = False

TOP_MOST = 10

DATA_DIR = '../data/'
DATA_FILES = [ 'a-3kittens.txt', 'bp-tom-kitten.txt', 'el-owl-cat.txt', 'rc-cat-fiddle.txt', 'rk-cat.txt' ]


#--
# cos_sim()
# computes and returns cosine similarity between two vectors j0 and j1 in termdoc matrix
#--
def cos_sim( j0, j1, num_t, termdoc ):
    #print 'j0: ',termdoc[j0][:]
    #print 'j1: ',termdoc[j1][:]
    sim_top = 0
    sim_bottom_0 = 0
    sim_bottom_1 = 0
    for t in range( num_t ):
        sim_top += termdoc[j0][t] * termdoc[j1][t]
        sim_bottom_0 += np.square( termdoc[j0][t] )
        sim_bottom_1 += np.square( termdoc[j1][t] )
    sim = sim_top / ( np.sqrt( sim_bottom_0 )) * ( np.sqrt( sim_bottom_1 ))
    return( sim )


#-----
# MAIN
#-----

# initalise list of dictionaries of most frequent words in each verse
freq_words = [ dict() for j in range( len( DATA_FILES )) ]

# get list of "stopwords"
print('fetching list of stopwords...')
stopwords = requests.get( "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt" ).content.decode('utf-8').split( "\n" )
print('number of stopwords = ' + str( len( stopwords )))
if ( DEBUGGING ):
    print('stopwords=', stopwords)

#-loop to read in the verses
for ( j, myfile ) in zip( range( len( DATA_FILES )), DATA_FILES ):
    with open( DATA_DIR+myfile ) as f:
        raw_verse = f.read()
    f.close()
    if ( DEBUGGING ):
        print('raw_verse=', raw_verse)
    print('file=', myfile)

    # initialise a TextBlob object using the verse
    # (this will decode any UTF-8 characters in the file)
    verse = textblob.TextBlob( raw_verse) 
    if ( DEBUGGING ):
        print('verse=', verse)

    # create a dictionary of words for this verse, removing the stopwords
    words = {}
    for w in verse.word_counts:
        if ( w not in stopwords ):
            words[w] = verse.word_counts[w]
    if ( DEBUGGING ):
        print(words)

    # sort the words in order to find the TOP_MOST most frequent
    sorted_words = sorted( words, key=words.__getitem__, reverse=True )
    for ( i, w ) in zip( range( TOP_MOST ), sorted_words ):
        freq_words[j][w] = verse.word_counts[w]
        print(i, w, verse.word_counts[w])

# now, the freq_words list contains a dictionary of the TOP_MOST most
# frequent words in each data file, and the word frequencies
if ( DEBUGGING ):
    print(freq_words)

# let's use this to create a term-document matrix
# start by getting a unique list of terms
terms = []
for j in range( len( DATA_FILES )):
    for w in freq_words[j]:
        if ( w not in terms ):
            terms.append( w )
if ( DEBUGGING ):
    print('terms=', terms)

# now we can use this to create a binary term-document matrix
termdoc = [[0 for t in range( len( terms ))] for j in range( len( DATA_FILES ))]
for j in range( len( DATA_FILES )):
    for t in range( len( terms )):
        if ( terms[t] in freq_words[j] ):
            termdoc[j][t] = freq_words[j][ terms[t] ]

# print term-document matrix
if ( DEBUGGING ):
    for t in range( len( terms )):
        print( terms[t], end='' )
    print()
    for j in range( len( DATA_FILES )):
        print( DATA_FILES[j], end='' )
        for t in range( len( terms )):
            print( termdoc[j][t], end='' )
        print()

# compute pairwise cosine similarity between document vectors
max_d  = cos_sim( 0, 1, len( terms ), termdoc )
max_j0 = 0
max_j1 = 1
for j0 in range( len( DATA_FILES )):
    for j1 in range( j0+1, len( DATA_FILES )):
        d = cos_sim( j0, j1, len( terms ), termdoc )
        print('cos similarity from {} to {} = {}'.format( j0, j1, d ))
        if ( d > max_d ):
            max_d = d
            max_j0 = j0
            max_j1 = j1
print('closest two verses by Cosine similarity are: {} ({}) and {} ({})'.format( DATA_FILES[max_j0], max_j0, DATA_FILES[max_j1], max_j1 ))
print('vectors=')
print(termdoc[max_j0][:])
print(termdoc[max_j1][:])
