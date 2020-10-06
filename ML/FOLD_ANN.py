# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:12:27 2020

@author: berke
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report



dataset_without_IMG_info = "datasets/Dataset without IMG info.xlsx"
dataset = pd.read_excel(dataset_without_IMG_info)

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

model = MLPClassifier(hidden_layer_sizes=(186, 186, 186, 186),
                      max_iter=120,
                      activation='relu',
                      solver='adam',
                      batch_size=60,
                      beta_1=0.88,
                      beta_2=0.988,
                      epsilon=1e-6,
                      alpha=0.000094,
                      learning_rate_init=0.000755)

    
counter = 1

folds = StratifiedKFold(n_splits=9)
for train, test in folds.split(X, y):
    print("############################")
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

model_path = "models/FOLD_mlp_clf.pkl"
joblib.dump(model, model_path)
    
