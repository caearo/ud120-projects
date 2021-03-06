#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


'''
print type(features_train), len(features_train), '*', len(features_train[0])
print type(features_test), len(features_test)
print type(labels_train), len(labels_train)
print type(labels_test), len(labels_test)
'''
#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score
from time import time

t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split = 40)
clf.fit(features_train, labels_train)
print 'fitting time:', time()-t0, 's'

#graphing
print 'graph...'
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree.pdf") 

t0 = time()
pred = clf.predict(features_test)
pred.unic
print 'prdictting time:', time()-t0, 's'
print type(pred), len(pred)

print accuracy_score(pred, labels_test)

#########################################################
#'''

