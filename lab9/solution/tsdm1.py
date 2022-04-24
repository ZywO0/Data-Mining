#--
# tsdm1.py
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

#-get a unique list of cameras
cameras = sorted( set( df['CAMERA.ID'].values ))

#-plot a time series for each camera
plt.figure()
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
    plt.plot( ts, viols, label=c, alpha=0.5 )
plt.xlabel( 'timestamp' )
plt.ylabel( 'number of violations' )
plt.legend()
plotfilename = '../plots/' + 'all-cameras.png'
plt.savefig( plotfilename )
plt.show()
plt.close()
