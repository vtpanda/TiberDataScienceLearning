#!/usr/local/bin/python3



from pandas import Series, DataFrame

import pandas as pd
import numpy as np
from sklearn import tree
#I had to install scipy to be able to use sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import graphviz

#read data in
df = pd.read_csv('~/Documents/GitHub/TiberDataScienceLearning/Data/Titanic/train.csv')



y = df[['Survived']]

x = df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

#just for testing
#x[x['Embarked'].isna()]
#just to show the missing ages
#x[x['Age'].isna()]
#still just playing around
#x.loc[:,'Age']

#turn missing ages into the mean age
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x.loc[:,['Age']])
x.loc[:,'Age'] = imputer.transform(x.loc[:,['Age']])

#x.groupby('Embarked').count()

#show na embarked
#x[x['Embarked'].isna()]
#61 829
#throw away two records that don't have embarcation
#x =  x[ x['Embarked'].notna()]

#found two ways of putting S into the NaN values for Embarked
#x['Embarked'] = df.Embarked.replace(np.NaN, 'S')
x.Embarked = x.Embarked.fillna('S')
#check to make sure that we did it correctly
#x.loc[[61, 829], :]

#encode string values
labelencoder_X = LabelEncoder()
x.loc[:,'Sex'] = labelencoder_X.fit_transform(x.loc[:,'Sex'])
x.loc[:,'Embarked'] = labelencoder_X.fit_transform(x.loc[:,'Embarked'])

#split off 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#train decision tree (??)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)


#i would like to figure out how to programmatically get the list of columns names and pass them in here,
#but we'll run with this for now
#why did True work for class_names and not ['Survived']? need the names of the output categories
#is it possible to turn the sex categories and embarked categories back to string for display purposes?
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                         class_names=['Did not survive', 'Survived'],
                         filled=True, rounded=True,
                         special_characters=True)

graph = graphviz.Source(dot_data)
#had to do "brew install graphviz" to get the following statement to work
#this was put one folder up; i manually put these files in the FirstLesson subfolder
graph.render("Titanic")

#predict values
y_predictions = clf.predict(x_test)
y_predictions_proba = clf.predict_proba(x_test)
