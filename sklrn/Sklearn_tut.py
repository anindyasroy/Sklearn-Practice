# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:15:04 2020

@author: Anindya
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()

#split it into featues and labels
X = iris.data
y = iris.target

classes = ['Irs Setosa', 'Iris VersiColor', 'Iris Virginica']

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model= svm.SVC()
model.fit(X_train, y_train)

print(model)

predictions = model.predict(X_test)
acc=accuracy_score(y_test, predictions)


print("Predctions : ", predictions);
print("Actual ; ", y_test)
print("Accuracy : ", acc)
