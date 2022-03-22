#--
# p2.py
# tokenizing and analysing text
# @author: letsios, sklar
# @created: 28 Jan 2021
#--

import textblob
import requests

DEBUGGING = False

TOP_MOST = 10

DATA_DIR  = '../data/'
DATA_FILE = 'el-owl-cat.txt'


#-read in the verse
with open( DATA_DIR + DATA_FILE ) as f:
    raw_verse = f.read()
f.close()
if ( DEBUGGING ):
    print('raw_verse=', raw_verse)

#-initialise a TextBlob object using the verse
# (this will decode any UTF-8 characters in the file)
# verse = textblob.TextBlob( raw_verse.decode( 'utf-8' )) 
verse = textblob.TextBlob( raw_verse) 
# if ( DEBUGGING ):
#     print('verse=',verse)
print('number of unique words = ', len( verse.words ))
# number of unique words =  221

#-sort the words in order to find the TOP_MOST most frequent
sorted_words = sorted( verse.word_counts, key=verse.word_counts.__getitem__, reverse=True )
print('{} most frequent words:'.format( TOP_MOST ))
for ( i, w ) in zip( range( TOP_MOST ), sorted_words ):
    print('{} = {}, {}'.format(( TOP_MOST-i) , w, verse.word_counts[w] ))

#-get a list of "stopwords" (words to skip over)
print('fetching list of stopwords...')
stopwords = requests.get( "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt" ).content.decode('utf-8').split( "\n" )
print('number of stopwords = ' + str( len( stopwords )))
if ( DEBUGGING ):
    print('stopwords=', stopwords)

#-create a dictionary of words for this verse, removing the stopwords
words = {}
for w in verse.word_counts:
    if ( w not in stopwords ):
        words[w] = verse.word_counts[w]
if ( DEBUGGING ):
    print(words)

# sort the words in order to find the TOP_MOST most frequent
sorted_words = sorted( words, key=words.__getitem__, reverse=True )
print('{} most frequent words, after removing stopwords:'.format( TOP_MOST ))
for ( i, w ) in zip( range( TOP_MOST ), sorted_words ):
    print('{} = {}, {}'.format(( TOP_MOST-i) , w, verse.word_counts[w] ))
