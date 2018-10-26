# TiberDataScienceLearning

The data sets are found at:

Titanic: https://www.kaggle.com/c/titanic/data

Telco: https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv

10/26/2018: Fixed some bugs.  Some of the other parameters are actually good, too.  Now to try some combinations.

10/25/2018: Tried a whole bunch of parameters.  The max_depth one seems to be the best one so far.  Perhaps some combination should be used.

Update on 10/18/2018: Tested different depths.

Update on 10/16/2018: Added a cleaned up version of my decision_tree_classifier.py script that I'll want to add some stuff to for improvement's sake.  Also added a first version of the second week's script.

Update #3: Okay, I'm now using median, but looking at the distribution of values for Age, there's very little difference between using mean and median; that said, I might as well keep that code around for future problems.  Also, I used Rachel's code as a pointer to get counts by Age, and I exported that data to Excel to view in graph form.  Next thing to learn is how to create graphs in Python so I don't have to export to Excel.  Also, at some point, I want to figure out how to create normalized values for Age as a code exercise.

Update #2: Okay, I figured out why I didn't get any deprecation warnings from using Imputer; I was using scikit-learn version 0.19.  I upgraded to scikit-learn 0.20.  I'm now using SimpleImputer

Update on 10/3/2018: So, reading the Python docuementation, it looks like the Imputer class is deprecated and will be removed in a future version.  Need to update that with something.  

Update on 9/30/2018: Fixed issue with submission file

Update on 9/29/2018: I've updated the decision_tree_classifier.py file to do preprocessing after splitting, and also to split the embarked field into three columns.  I'm also outputting a file for submission.

The decision_tree_classifier.py file is the main file I'm working with to create and use the decision tree.

The read_in_data_and_query.py file was just me playing around with selecting columns, filtering, etc etc etc.
