#!/usr/local/bin/python3

#this does item number 3 in the first lesson


from pandas import Series, DataFrame
import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt

#read data in
df = pd.read_csv('~/Documents/GitHub/TiberDataScienceLearning/Data/Titanic/train.csv')

#show all data-ish
df

#show and display counts
df['Age'].value_counts()
#thing = df['Sex'].value_counts()


#using plotly
grouped = df.groupby('Sex')['PassengerId'].count()
data = [go.Bar(
    x=grouped.index,
    y=grouped.values
)]
layout = go.Layout(
    title='Counts by Sex'
)
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig, filename='sex.html')

#using pyplot
plt.bar(grouped.index, grouped.values)
plt.title('Counts by Sex')
plt.show()

#show the Name column
df['Name']

#show first three rows, i think?
df[:3]

#show me all people that survived
df[df['Survived'] == 1]

#show me the row number 2; got a deprecation warning
df.ix[[2],]

#show me the rows number 2 and 4 but only the Survived and Name columns;
#presumably [2,4] is a label rather than positional, though i didn't test this out;
#got a deprecation warning
df.ix[[2,4],['Survived','Name']]

#same as above using label indexing; no deprecation warning
df.loc[[2,4], ['Survived','Name']]

#same as above using positional indexing;
#note that the labels happen to be the same as the positions
df.iloc[[2,4], [1,3]]

#get entire records using label and positional indexing
df.loc[[2,4],]
df.iloc[[2,4], ]

#okay; this works
df.loc[ : , ['Survived','Name']]
df.iloc[ : , [1,3]]
#this also work
df[['Survived','Name']]

#more ways of doing things (though it's unclear why you would want to do this)
#convert positional index to label index (right?)
df.loc[df.index[[0, 2]], 'Survived']
#convert label index to positional index
df.iloc[[0, 2], df.columns.get_loc('Survived')]
#etc etc etc
df.loc[df.index[[0, 2]], ['Survived', 'Name']]
df.iloc[[0, 2], df.columns.get_indexer(['Survived', 'Name'])]
