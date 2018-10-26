#!/usr/local/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preprocessdataframe (df):
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imputer = imputer.fit(df.loc[:,['Age']])
    df.loc[:,'Age'] = imputer.transform(df.loc[:,['Age']])

    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imputer = imputer.fit(df.loc[:,['Fare']])
    df.loc[:,'Fare'] = imputer.transform(df.loc[:,['Fare']])

    df.Embarked = df.Embarked.fillna('S')

    df = pd.get_dummies(data=df, columns=['Embarked', 'Pclass', 'Sex'])

    return df




df = pd.read_csv('~/Documents/GitHub/TiberDataScienceLearning/Data/Titanic/train.csv')
y = df[['Survived']]
x = df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
x_train = preprocessdataframe(x_train)
x_test = preprocessdataframe(x_test)

#basic decision tree with no hyperparameters
clf_basic = tree.DecisionTreeClassifier()
cross_val_roc = cross_val_score(clf_basic, X=x_train, y=y_train, cv=10, scoring='roc_auc')
roc_score = np.mean(cross_val_roc)
print("No hyperparameter decision tree: ", roc_score)

#testing out max_depth parameters with values from 1 to 12
aucs = dict()
for i in range(1,12):
    clf_maxdepth = tree.DecisionTreeClassifier(max_depth=i)
    cross_val_roc = cross_val_score(clf_maxdepth, x_train, y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the max_depth hyperparameter: ", aucs)

aucs = dict()
params = [.01, .05, .1, .2, .5]
for i in params:
    clf_min_samples_split = tree.DecisionTreeClassifier(min_samples_split = i)
    cross_val_roc = cross_val_score(clf_min_samples_split, X=x_train, y=y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the min_samples_split hyperparameter: ", aucs)

aucs = dict()
params = [.01, .05, .1, .2, .5]
for i in params:
    clf_min_samples_leaf = tree.DecisionTreeClassifier(min_samples_leaf = i)
    cross_val_roc = cross_val_score(clf_min_samples_leaf, X=x_train, y=y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the min_samples_leaf hyperparameter: ", aucs)

aucs = dict()
for i in range(1,8):
    clf_maxfeatures = tree.DecisionTreeClassifier(max_features=i)
    cross_val_roc = cross_val_score(clf_maxfeatures, x_train, y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the max_features hyperparameter: ", aucs)

aucs = dict()
params = [.001, .0001, .01, .05, .1, .2, .5]
for i in params:
    clf_min_impurity_decrease = tree.DecisionTreeClassifier(min_impurity_decrease = i)
    cross_val_roc = cross_val_score(clf_min_impurity_decrease, X=x_train, y=y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the min_impurity_decrease hyperparameter: ", aucs)
