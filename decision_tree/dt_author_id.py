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
from time import time
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier(min_samples_split=40)
t = time()
clf = clf.fit(features_train, labels_train)
print "Training Time: ", round(time()-t, 3), "s"
t = time()
labels_predicted = clf.predict(features_test)
print "Prediction Time: ", round(time()-t, 3), "s"
print "Accuracy Score: ", accuracy_score(labels_test, labels_predicted)
print "Features len: ", len(features_train[0])



