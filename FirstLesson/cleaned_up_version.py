#!/usr/local/bin/python3

import pandas as pd
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import graphviz
import matplotlib.pyplot as plt
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

def showmultiroccurve(params): #this should be a list of dictionaries of fpr, tpr, and roc_auc
    plt.figure()
    lw = 2
    for param in params:
        plt.plot(param["fpr"], param["tpr"], color='darkorange',
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

#split off 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
x_train = preprocessdataframe(x_train)
x_test = preprocessdataframe(x_test)


clf = tree.DecisionTreeClassifier()
model = clf.fit(x_train, y_train)

showdecisiontree(model, x_train.columns, "Documents/GitHub/TiberDataScienceLearning/FirstLesson/TitanicDecisionTree")
y_predictions = model.predict(x_test)
y_predictions_proba = model.predict_proba(x_test)

cm = pd.DataFrame(
    confusion_matrix(y_test, y_predictions),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
)

score = model.score(x_test, y_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predictions)
roc_auc = metrics.roc_auc_score(y_test, y_predictions)
label = 'No Hyperparameters'
#the following was what I originally had, but I pulled the above from Rachael's code because it's betters
#roc_auc2 = metrics.auc(fpr, tpr)
showroccurve(fpr, tpr, roc_auc, label)

cross_val_roc = cross_val_score(model, X=x_train, y=y_train, cv=10, scoring='roc_auc')
cross_val_mean_roc_score = np.mean(cross_val_roc)


#10-fold
kf = KFold(n_splits=10)
kf.get_n_splits(x)

clfkfold = tree.DecisionTreeClassifier()

for train_index, test_index in kf.split(x_train, y_train):
    cvv_x_train, cvv_x_test = x_train.iloc[train_index], x_train.iloc[test_index]
    cvv_y_train, cvv_y_test = y_train.iloc[train_index], y_train.iloc[test_index]

    # cvv_x_train = preprocessdataframe(cvv_x_train)
    # cvv_x_test = preprocessdataframe(cvv_x_test)

    clfkfold = clfkfold.fit(cvv_x_train, cvv_y_train)
    cvv_y_predictions = clfkfold.predict(cvv_x_test)

    cvv_cm = pd.DataFrame(
        confusion_matrix(cvv_y_test, cvv_y_predictions),
        columns=['Predicted Not Survival', 'Predicted Survival'],
        index=['True Not Survival', 'True Survival']
    )
    cvv_score = clfkfold.score(cvv_x_test, cvv_y_test)

    cvv_fpr, cvv_tpr, thresholds = metrics.roc_curve(cvv_y_test, cvv_y_predictions)
    cvv_roc_auc = metrics.auc(cvv_fpr, cvv_tpr)
