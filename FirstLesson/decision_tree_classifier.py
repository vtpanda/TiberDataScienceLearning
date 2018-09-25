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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
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

#is the following a better alternative?
#df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})





#10-fold
kf = KFold(n_splits=10)
kf.get_n_splits(x)

clfkfold = tree.DecisionTreeClassifier()

for train_index, test_index in kf.split(x):
    #i'm not sure if iloc (positional) or loc (Label) is better here; does split return positions or labels?
    cvv_x_train, cvv_x_test = x.iloc[train_index], x.iloc[test_index]
    cvv_y_train, cvv_y_test = y.iloc[train_index], y.iloc[test_index]
    #this is where I've left off; i guess i need to train each set and get the results, right?
    clfkfold = clfkfold.fit(cvv_x_train, cvv_y_train)
    cvv_y_predictions = clfkfold.predict(cvv_x_test)
    #confusion matrix
    cm = pd.DataFrame(
        confusion_matrix(cvv_y_test, cvv_y_predictions),
        columns=['Predicted Not Survival', 'Predicted Survival'],
        index=['True Not Survival', 'True Survival']
    )
    print ("True Positive Rate: " , cm.loc['True Survival', 'Predicted Survival'] / (cm.loc['True Survival', 'Predicted Survival'] + cm.loc['True Survival', 'Predicted Not Survival']))
    print ("False Positive Rate: " , cm.loc['True Not Survival', 'Predicted Survival'] / (cm.loc['True Not Survival', 'Predicted Survival'] + cm.loc['True Not Survival', 'Predicted Not Survival']))
    print("Overall Score:",  clfkfold.score(cvv_x_test, cvv_y_test))
    #print (cm)

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

#do 10-fold cross-validation;
#this seems to score for both positive and negative rather than just for positive, so I'm not going to use it
#scores = cross_val_score(clf, X=x, y=y, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#get my score of held out data
myscore = clf.score(x_test, y_test) #looks roughly similar;

#predict values
y_predictions = clf.predict(x_test)
y_predictions_proba = clf.predict_proba(x_test)

cm = pd.DataFrame(
    confusion_matrix(y_test, y_predictions),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
)

print ("True Positive Rate: " , cm.loc['True Survival', 'Predicted Survival'] / (cm.loc['True Survival', 'Predicted Survival'] + cm.loc['True Survival', 'Predicted Not Survival']))
print ("False Positive Rate: " , cm.loc['True Not Survival', 'Predicted Survival'] / (cm.loc['True Not Survival', 'Predicted Survival'] + cm.loc['True Not Survival', 'Predicted Not Survival']))
print("Overall Score:",  clf.score(x_test, y_test))
print (cm)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predictions)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
