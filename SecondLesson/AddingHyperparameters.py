#!/usr/local/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from time import time

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

def showdecisiontree(model, feature_names, name):
    dot_data = tree.export_graphviz(model, out_file=None,
         feature_names=feature_names,
         class_names=['Did not survive', 'Survived'],
         filled=True, rounded=True,
         special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(name)
    return True

def showroccurve(fpr, tpr, roc_auc, label):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve - {0} (area = {1:0.2f})'.format(label, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def showmultiroccurve(params): #this should be a list of dictionaries of fpr, tpr, roc_auc, label, and color
    plt.figure()
    lw = 2
    for param in params:
        plt.plot(param["fpr"], param["tpr"], color=param["color"],
             lw=lw, label='ROC curve - {0} (area = {1:0.2f})'.format(param["label"],param["roc_auc"]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


df = pd.read_csv('~/Documents/GitHub/TiberDataScienceLearning/Data/Titanic/train.csv')
y = df[['Survived']]
x = df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
x_train = preprocessdataframe(x_train)
x_test = preprocessdataframe(x_test)

#basic decision tree with no hyperparameters
clf = tree.DecisionTreeClassifier()
cross_val_roc = cross_val_score(clf, X=x_train, y=y_train, cv=10, scoring='roc_auc')
roc_score = np.mean(cross_val_roc)
print("No hyperparameter decision tree: ", roc_score)
#output: No hyperparameter decision tree:  0.754228329809725


#testing out max_depth parameters with values from 1 to 12
aucs = dict()
for i in range(1,12):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    cross_val_roc = cross_val_score(clf, x_train, y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the max_depth hyperparameter: ", aucs)
#output: Using the max_depth hyperparameter:  {1: 0.7652214280121256, 2: 0.8065079085427922, 3: 0.8418283751132588, 4: 0.8427579714643668, 5: 0.8424975810150229, 6: 0.8231955490676419, 7: 0.802657108180364, 8: 0.80016373591955, 9: 0.793288190599237, 10: 0.7831202039777623, 11: 0.7548523720035347}

#test out min_samples_split parameter with values of [.01, .05, .1, .2, .5]
aucs = dict()
params = [.01, .05, .1, .2, .5]
for i in params:
    clf = tree.DecisionTreeClassifier(min_samples_split = i)
    cross_val_roc = cross_val_score(clf, X=x_train, y=y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the min_samples_split hyperparameter: ", aucs)
#output: Using the min_samples_split hyperparameter:  {0.01: 0.789461929650883, 0.05: 0.8164925836437463, 0.1: 0.8302917044196114, 0.2: 0.8400290627761556, 0.5: 0.804992337550477}

#test out min_samples_leaf parameter with values of [.01, .05, .1, .2, .5]
aucs = dict()
params = [.01, .05, .1, .2, .5]
for i in params:
    clf = tree.DecisionTreeClassifier(min_samples_leaf = i)
    cross_val_roc = cross_val_score(clf, X=x_train, y=y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the min_samples_leaf hyperparameter: ", aucs)
#output: Using the min_samples_leaf hyperparameter:  {0.01: 0.8264754410103248, 0.05: 0.8296723463874626, 0.1: 0.8305953709296732, 0.2: 0.7927730726422586, 0.5: 0.5233987158405763}


#test out max_features parameter with values of 1 to 8
aucs = dict()
for i in range(1,8):
    clf = tree.DecisionTreeClassifier(max_features=i)
    cross_val_roc = cross_val_score(clf, x_train, y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the max_features hyperparameter: ", aucs)
#output: Using the max_features hyperparameter:  {1: 0.7391151968186852, 2: 0.7358194696130742, 3: 0.7374462020537602, 4: 0.7392733746658164, 5: 0.7444237502377037, 6: 0.7732762984216472, 7: 0.7447708326342048}

#test out min_impurity_decrease parameter with values of [.0001, .001, .01, .05, .1, .2, .5]
aucs = dict()
params = [.0001, .001, .01, .05, .1, .2, .5]
for i in params:
    clf = tree.DecisionTreeClassifier(min_impurity_decrease = i)
    cross_val_roc = cross_val_score(clf, X=x_train, y=y_train, cv=10, scoring='roc_auc')
    roc_score = np.mean(cross_val_roc)
    aucs[i] = roc_score
print("Using the min_impurity_decrease hyperparameter: ", aucs)
#output: Using the min_impurity_decrease hyperparameter:  {0.001: 0.7608536220454825, 0.0001: 0.756113390270367, 0.01: 0.8196643903039252, 0.05: 0.7652214280121256, 0.1: 0.7652214280121256, 0.2: 0.5, 0.5: 0.5}

#use GridSearchCV to find the best hyperparameters
param_grid = [
  {'max_depth': range(1,12), 'min_samples_split': [.01, .05, .1, .2, .5], 'min_samples_leaf': [.01, .05, .1, .2, .5], 'max_features': range(1,8), 'min_impurity_decrease': [.001, .0001, .01, .05, .1, .2, .5]},
 ]
clf = tree.DecisionTreeClassifier()
gscv = GridSearchCV(clf, param_grid, cv=10, scoring='roc_auc')
start = time()
gscv = gscv.fit(x_train , y_train)
stop = time()
print("Best Score: ", gscv.best_score_)
print("Best Parameters: ", gscv.best_params_)
print("Time: ", stop-start)
# Best Score:  0.8550049218653869
# Best Parameters:  {'max_depth': 7, 'max_features': 4, 'min_impurity_decrease': 0.0001, 'min_samples_leaf': 0.01, 'min_samples_split': 0.05}
# Time:  667.2244110107422

#use RandomizedSearchCV to find the best hyperparameters
param_grid = {'max_depth': range(1,12), 'min_samples_split': [.01, .05, .1, .2, .5], 'min_samples_leaf': [.01, .05, .1, .2, .5], 'max_features': range(1,8), 'min_impurity_decrease': [.001, .0001, .01, .05, .1, .2, .5]}
clf = tree.DecisionTreeClassifier()
gscv = RandomizedSearchCV(clf, param_grid, cv=10, scoring='roc_auc')
start = time()
gscv = gscv.fit(x_train , y_train)
stop = time()
print("Best Score: ", gscv.best_score_)
print("Best Parameters: ", gscv.best_params_)
print("Time: ", stop-start)
# Best Score:  0.8372774909308094
# Best Parameters:  {'min_samples_split': 0.05, 'min_samples_leaf': 0.01, 'min_impurity_decrease': 0.0001, 'max_features': 6, 'max_depth': 7}
# Time:  0.5015101432800293
