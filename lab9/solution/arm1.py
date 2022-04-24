#--
# arm1.py
# association rule mining -- plots a graph of the rules
# @author: letsios, sklar
# @created: 28 Jan 2021
#
#--

import sys
import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

DEBUGGING = True
DATA_DIR  = '../data/'
DATA_FILE = 'survey-data.csv'

MINIMUM_COVERAGE = 500
MINIMUM_CONFIDENCE = 0.5


#-define a class for storing the antedecent and consequent clauses
class Clause:
    def __init__( self, column0, value0 ):
        self.column = column0
        self.value = value0
    def __str__( self ):
        return( '(' + self.column + '==' + self.value + ')' )
    def getColumn( self ):
        return( self.column )
    def getValue( self ):
        return( self.value )

#-define a class for storing association rules
class AssocRule:
    def __init__( self, ant_column, ant_value, con_column, con_value ):
        self.antecedent = Clause( ant_column, ant_value )
        self.consequent = Clause( con_column, con_value )
    def __str__( self ):
        return( str( self.antecedent ) + '->' + str( self.consequent ))
    def getAntecedent( self ):
        return( self.antecedent )
    def getConsequent( self ):
        return( self.consequent )


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

#-find all one-item sets and calculate their coverage (number of
# transactions where the item set occurs). store the coverage values
# in a dictionary of dictionaries.
coverage1 = {}
for c in df.columns:
    coverage1[c] = {}
# loop through instances to create one-item sets
for j in range( M ):
    # loop through the attributes for this instance
    for c in df.columns:
        if ( df[c].dtypes == 'object' ): # skip the "Unnamed"
                                         # attribute, which is a
                                         # unique instance identifier
                                         # and won't help with the
                                         # association rule building
            att_value = df[c].values[j]
            if ( att_value in coverage1[c].keys() ):
                coverage1[c][att_value] += 1
            else:
                coverage1[c][att_value] = 1
# remove one-item sets that do not meet the minimum coverage
num_item1 = 0
for c in df.columns:
    for k in list(coverage1[c].keys()):
        if ( coverage1[c][k] < MINIMUM_COVERAGE ):
            del coverage1[c][k]
        else:
            num_item1 += 1

# to make the code clearer below, make a list of all the columns in the one-item sets
columns1 = []
for c in df.columns:
    if ( len( coverage1[c] ) > 0 ):
        columns1.append( c )

#-find all two-item sets by making pairs of the one-item sets that are
# above the minimum coverage value
coverage2 = {}
# initialise a dictionary of columns for two-item sets
for c1 in columns1:
    coverage2[c1] = {}

# loop through instances to create two-item sets
for j in range( M ):
    # loop through list of column names using a numeric index to make
    # sure we don't have duplicates and are just creating the
    # triangular matrix of combinations
    for i1 in range( len( columns1 )):
        c1 = columns1[i1]
        att1_value = df[c1].values[j]
        if ( att1_value not in coverage2[c1].keys() ):
            coverage2[c1][att1_value] = {}
        for i2 in range( i1+1, len( columns1 )):
            c2 = columns1[i2]
            if ( c2 not in coverage2[c1][att1_value].keys() ):
                coverage2[c1][att1_value][c2] = {}
            att2_value = df[c2].values[j]
            if ( att2_value not in coverage2[c1][att1_value][c2].keys() ):
                coverage2[c1][att1_value][c2][att2_value] = 1
            else:
                coverage2[c1][att1_value][c2][att2_value] += 1

# remove two-item sets that do not meet the minimum coverage
num_item2 = 0
for c1 in columns1:
    for k1 in list(coverage2[c1].keys()):
        for k2 in list(coverage2[c1][k1].keys()):
            for k3 in list(coverage2[c1][k1][k2].keys()):
                if ( coverage2[c1][k1][k2][k3] < MINIMUM_COVERAGE ):
                    del coverage2[c1][k1][k2][k3]
                else:
                    num_item2 += 1

#-produce two association rules for each two-item set, putting each of
# the items, in turn, in the antecedent and the other in the
# consequent
rules = []
for c1 in columns1:
    for k1 in coverage2[c1].keys():
        for k2 in coverage2[c1][k1].keys():
            for k3 in coverage2[c1][k1][k2].keys():
                rules.append( AssocRule( c1, k1, k2, k3 ))
                rules.append( AssocRule( k2, k3, c1, k1 ))

#-for each rule, evaluate the confidence and retain the rule if it
# meets the minimum requirement for confidence.
# confidence is the coverage count for the rule / coverage count for
# the antecedent.
confidence = [0.0 for i in range( len( rules ))]
for ( i, r ) in zip( range( len( rules )), rules ):
    ant = r.getAntecedent()
    con = r.getConsequent()
    rule_coverage = 0
    ant_coverage = 0
    for j in range( M ):
        # compute the coverage count for the antecedent
        if ( df[ ant.getColumn() ].values[j] == ant.getValue() ):
            ant_coverage += 1
            # compute the coverage count for the rule (note that we
            # only need to look for the whole rule count if the
            # antecedent is true)
            if ( df[ con.getColumn() ].values[j] == con.getValue() ):
                rule_coverage += 1
    confidence[i] = float( rule_coverage ) / float( ant_coverage )

#-generate a list of rules for plotting an undirected graph of the rules network
list_of_rules = []
for r in rules:
    list_of_rules.append(( str( r.antecedent ), str( r.consequent ) ))
if DEBUGGING:
    print(list_of_rules)

# draw the graph
plt.figure()
g = nx.Graph()
g.add_edges_from( list_of_rules )
nx.draw( g, node_size=3000, with_labels=True, node_color='lightgrey' )
plt.savefig( '../plots/arm1-undirected.png' )
plt.show()
