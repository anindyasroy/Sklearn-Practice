# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:44:02 2020

@author: Anindya
"""
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

bc = load_breast_cancer()

print(bc)

X = bc.data
print(X)

y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2)

model =KMeans(n_clusters=2, random_state=0)

model.fit(X_train)

predictions= model.predict(X_test)

labels =model.labels_

print("labels :", labels)
print("predictions", predictions)
print("Accuracy :", accuracy_score(y_test, predictions))
print("Actual values : ", y_test )

