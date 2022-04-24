#--
# tsdm0.py
# time series data mining
# @author: letsios, sklar
# @created: 28 Jan 2021
#
#--

import sys
import csv
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

DEBUGGING = True
PLOTTING = True

DATA_DIR  = '../data/'
DATA_FILE = 'traffic-data.csv'


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
    print('number of attributes = ' + str( N ))
    print('number of instances = ' + str( M ))
    for ( i, c, t ) in zip( range( N ), df.columns, df.dtypes ):
        print('{} - {} ({})'.format( i, c, t ))
# the above reports the following:
# number of attributes = 10
# number of instances = 5426
#  0 - Unnamed: 0 (int64)
#  1 - DATE (object)
#  2 - CAMERA.ID (object)
#  3 - ADDRESS (object)
#  4 - VIOLATIONS (int64)
#  5 - LATITUDE (float64)
#  6 - LONGITUDE (float64)
#  7 - LOCATION (object)
#  8 - date (object)
#  9 - count (int64)

#-get a unique list of cameras
cameras = sorted( set( df['CAMERA.ID'].values ))
print('cameras = ', cameras)

#-count the number of points per camera
for c in cameras:
    num = len( df.loc[ ( df['CAMERA.ID']==c ) ] )
    print('camera: ', c, ' incidents=', num)
    
#-count the number of violations per camera
total_violations = 0
for c in cameras:
    d = pd.DataFrame( df.loc[ ( df['CAMERA.ID']==c ) ] )
    v = np.array( d['VIOLATIONS'].values )
    print('camera: ', c, ' incidents=', len( v ), ' violations=', np.sum( v ))
    total_violations += np.sum( v )
print('total number of violations=', total_violations)

#-get the earliest and latest dates in the data set
d = np.array( df['date'].values )
min_date = np.min( d )
max_date = np.max( d )
( yyyy, mm, dd ) = min_date.split( '-' )
tm = time.struct_time(( int( yyyy ), int( mm ), int( dd ), 0, 0, 0, 0, 0, 0 ))
min_ts = time.mktime( tm )
( yyyy, mm, dd ) = max_date.split( '-' )
tm = time.struct_time(( int( yyyy ), int( mm ), int( dd ), 0, 0, 0, 0, 0, 0 ))
max_ts = time.mktime( tm )
print('earliest time=', min_date, min_ts)
print('latest time=', max_date, max_ts)

#-plot a time series for each camera
if PLOTTING:
    for c in cameras:
        ts = []
        viols = []
        d = pd.DataFrame( df.loc[ ( df['CAMERA.ID']==c ) ] )
        d = d.sort_values( by=['date'] )
        for j in range( len( d )):
            ( yyyy, mm, dd ) = d['date'].values[j].split( '-' )
            tm = time.struct_time(( int( yyyy ), int( mm ), int( dd ), 0, 0, 0, 0, 0, 0 ))
            ts.append( time.mktime( tm ))
            viols.append( float( d['VIOLATIONS'].values[j] ))
        plt.figure()
        plt.plot( ts, viols )
        plt.xlabel( 'timestamp' )
        plt.ylabel( 'number of violations' )
        plotfilename = '../plots/' + 'camera-' + c + '-ts.png'
        plt.savefig( plotfilename )
        plt.show()
        plt.close()
