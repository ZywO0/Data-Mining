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
import numpy as np
import matplotlib.pyplot as plt

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
    print('number of attributes = ' + str( N ))
    print('number of instances = ' + str( M ))
    for ( i, c, t ) in zip( range( N ), df.columns, df.dtypes ):
        print('{} - {} ({})'.format( i, c, t ))
# the above reports the following:
# number of attributes = 13
# number of instances = 51
#  0 - Code (object)
#  1 - Area (object)
#  2 - Births (object)
#  3 - 1 Year Survival Numbers (object)
#  4 - 1 Year Survival Percent (int64)
#  5 - 2 Year Survival Numbers (object)
#  6 - 2 Year Survival Percent (int64)
#  7 - 3 Year Survival Numbers (object)
#  8 - 3 Year Survival Percent (int64)
#  9 - 4 Year Survival Numbers (object)
# 10 - 4 Year Survival Percent (int64)
# 11 - 5 Year Survival Numbers (object)
# 12 - 5 Year Survival Percent (int64)

# put the survival data in an array
sdata = [[0 for i in range( M )] for i in range( 5 )]
for j in range( M ):
    sdata[0][j] = df['1 Year Survival Percent'].values[j]
    sdata[1][j] = df['2 Year Survival Percent'].values[j]
    sdata[2][j] = df['3 Year Survival Percent'].values[j]
    sdata[3][j] = df['4 Year Survival Percent'].values[j]
    sdata[4][j] = df['5 Year Survival Percent'].values[j]

# compute the average survival over all regions
pct = [0 for i in range( 6 )]
pct[0] = 100
for i in range( 5 ):
    pct[i+1] = np.mean( sdata[i][:] )

# plot the average survival over all regions
plt.figure()
plt.plot( np.linspace( 0, 5, 6 ), pct )
plt.xlim(( 0, 5 ))
plt.ylim(( 0, 100 ))
plt.ylabel( 'percent (%)' )
plt.xlabel( 'survival time (years)' )
plt.savefig( '../plots/survival-mean.png' )
plt.show()
plt.close()

# compute the survival for the first region in the data set (City of London)
pct = [0 for i in range( 6 )]
pct[0] = 100
for i in range( 5 ):
    pct[i+1] = sdata[i][0]

# plot the survival curve for the first region in the data set
plt.figure()
plt.plot( np.linspace( 0, 5, 6 ), pct )
plt.xlim(( 0, 5 ))
plt.ylim(( 0, 100 ))
plt.ylabel( 'percent (%)' )
plt.xlabel( 'survival time (years)' )
plt.savefig( '../plots/survival-city.png' )
plt.show()
plt.close()

# plot the average survival over all regions
plt.figure()
for j in range( M ):
    pct = [0 for i in range( 6 )]
    pct[0] = 100
    for i in range( 5 ):
        pct[i+1] = sdata[i][j]
    plt.plot( np.linspace( 0, 5, 6 ), pct )
plt.xlim(( 0, 5 ))
plt.ylim(( 0, 100 ))
plt.ylabel( 'percent (%)' )
plt.xlabel( 'survival time (years)' )
plt.savefig( '../plots/survival-all.png' )
plt.show()
plt.close()


# In which region is a business most likely to fail after 5 years?
i = np.argmin( df['5 Year Survival Percent'].values )
print('In which region is a business most likely to fail after 5 years?')
print(df['Area'].values[i], end=' ')
print('with a rate of ', end=' ')
print(df['5 Year Survival Percent'].values[i], end=' ')
print('%')

# In which region is a business most likely to survive after 5 years?
i = np.argmax( df['5 Year Survival Percent'].values )
print('In which region is a business most likely to survive after 5 years?')
print(df['Area'].values[i], end=' ')
print('with a rate of ', end=' ')
print(df['5 Year Survival Percent'].values[i], end=' ')
print('%')


