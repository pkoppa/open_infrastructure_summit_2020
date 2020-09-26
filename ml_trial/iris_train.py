#!/usr/bin/env python3

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

#Loading Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=75, test_size=0.75)


#Build and Training using  Support Vector Classifier
model_svc = svm.SVC(gamma='auto')
model_svc.fit(X_train, y_train)

#Finding accuracy of model
accuracy_svc = model_svc.score(X_test, y_test)
print("Accuracy for Support Vector Classifier is: {}".format(accuracy_svc))


#Build and Training using RandomForestClassifier
model_rfc = RandomForestClassifier(n_estimators=25)
model_rfc.fit(X_train, y_train)

#Finding the accuracy of model
accuracy_rfc = model_rfc.score(X_test,y_test)
print("Accuracy for Support Vector Classifier is: {}".format(accuracy_rfc))



with open('model_svc.pkl','wb') as model_svc_pickle:
    pickle.dump(model_svc, model_svc_pickle)

with open('model_rfc.pkl','wb') as model_rfc_pickle:
    pickle.dump(model_rfc, model_rfc_pickle)

