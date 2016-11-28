#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




# your code goes here
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# These lines effectively slice the training dataset down to 1% of its original size,
# tossing out 99% of the training data
# Uncomment to see it in action

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

svc = None
# linear
# svc = SVC(kernel='linear')
# rbf
svc = SVC(kernel='rbf', C=10000.0)
t = time.time()
svc.fit(features_train, labels_train)
print "Training Time: ", round(time.time()-t, 3), "s"
print "Saving Model.."
file_name = 'finalized_model.sav'
joblib.dump(svc, file_name)
time.sleep(10)
svc = joblib.load(file_name)
t = time.time()
labels_predicted = svc.predict(features_test)
print "Prediction Time: ", round(time.time()-t, 3), "s"
print "Accuracy Score: ", accuracy_score(labels_test, labels_predicted)
print "Test 10 : ", svc.predict(features_test[10])
print "Test 26 : ", svc.predict(features_test[26])
print "Test 50: ", svc.predict(features_test[50])

count = 0
for _ in labels_predicted:
    if _ == 1:
        count += 1

print "Chris Count: ", count

