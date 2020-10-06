# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:18:41 2020

@author: berke
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report

dataset_without_IMG_info = "datasets/Dataset without IMG info.xlsx"
dataset = pd.read_excel(dataset_without_IMG_info)


X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

model = KNeighborsClassifier(n_neighbors=1)

folds = StratifiedKFold(n_splits=9)

counter = 1

for train, test in folds.split(X, y):
    
    print("##############################")
    print("FOLD = ", counter)
    
    x_train = X[train]
    x_test = X[test]
    y_train = y[train]
    y_test = y[test]
    
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print(classification_report(y_test, preds))
    
    counter += 1

from sklearn.externals import joblib

model_path = "models/FOLD_knn_clf.pkl"
joblib.dump(model, model_path)
