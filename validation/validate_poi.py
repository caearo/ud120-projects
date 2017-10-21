#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
### L17
from sklearn import tree 
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print "Accuracy of L17(overfit):",accuracy_score(clf.predict(features), labels)

### L28
from sklearn.cross_validation import train_test_split
f_train, f_test, l_train, l_test = \
train_test_split(features, labels, test_size = .3, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf.fit(f_train, l_train)
print "Accuracy of L18:",accuracy_score(clf.predict(f_test), l_test)
