#--
# sdm-data0.py
# survival data mining data exploration using downloaded data
# @author: letsios, sklar
# @created: 28 Jan 2021
#
#--

import sys
import csv
import pandas as pd

DEBUGGING = True
DATA_DIR  = '../data/'
DATA_FILE = 'business-survival.csv'


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
M = len( df.values )

#-print columns
if DEBUGGING:
    print('INPUT FILE = ' + DATA_DIR + DATA_FILE)
    print('number of attributes = %d' + str( N ))
    print('number of instances = %d' + str( M ))
    for ( i, c, t ) in zip( range( N ), df.columns, df.dtypes ):
        print('{} - {} ({})'.format( i, c, t ))
