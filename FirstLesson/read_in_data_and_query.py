#!/usr/local/bin/python3

#this does item number 3 in the first lesson


from pandas import Series, DataFrame

import pandas as pd

#read data in
df = pd.read_csv('Data/Titanic/train.csv')

#show all data-ish
df

#show the Name column
df['Name']

#show first three rows, i think?
df[:3]

#show me all people that survived
df[df['Survived'] == 1]

#show me the row number 2; got a deprecation warning
df.ix[[2],]

#show me the rows number 2 and 4 but only the Survived and Name columns;
#i suspect that the [2,4] is positional rather than label;
#got a deprecation warning
df.ix[[2,4],['Survived','Name']]

#same as above using label indexing
df.loc[[2,4], ['Survived','Name']]

#same as above using positional indexing;
#note that the labels happen to be the same as the positions
df.iloc[[2,4], [1,3]]

#get entire records using label and positional indexing
df.loc[[2,4],]
df.iloc[[2,4], ]

#note that these don't seem to work; perhaps try a few more things
df.loc[ , ['Survived','Name']]
df.iloc[ , [1,3]]
#this does work however
df[['Survived','Name']]

#more ways of doing things (though it's unclear why you would want to do this)
#convert positional index to label index (right?)
df.loc[df.index[[0, 2]], 'Survived']
#convert label index to positional index
df.iloc[[0, 2], df.columns.get_loc('Survived')]
#etc etc etc
df.loc[df.index[[0, 2]], ['Survived', 'Name']]
df.iloc[[0, 2], df.columns.get_indexer(['Survived', 'Name'])]
