# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:59:32 2020

@author: berke
"""

from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report
import numpy as np
from sklearn.utils import shuffle

dataset_without_IMG_info = "datasets/Dataset without IMG info.xlsx"
dataset = pd.read_excel(dataset_without_IMG_info)

dataset = shuffle(dataset)

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values


attr_count = X.shape[1]

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

classes = []
for i in range(len(y)):
    if y[i] not in classes:
        classes.append(y[i])

OL_neuron_count = len(classes)

model = keras.models.Sequential()
model.add(keras.layers.Dense(200, input_dim=attr_count, activation=keras.activations.relu))
model.add(keras.layers.Dense(200, activation=keras.activations.relu))
model.add(keras.layers.Dense(OL_neuron_count, activation=keras.activations.softmax))

model.compile(optimizer=keras.optimizers.Adamax(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

folds = StratifiedKFold(n_splits=9)
counter = 1
for train, test in folds.split(X, encoded_y):
    print("##########################")
    print("FOLD = ", counter)

    x_train = X[train]
    x_test = X[test]
    y_train = encoded_y[train]
    y_test = encoded_y[test]
    
    model.fit(x_train, y_train, epochs=120, batch_size=80, verbose=0)
    preds = model.predict([x_test])
    
    cnv_preds = []
    for i in range(len(y_test)):
        cnv_preds.append(np.argmax(preds[i]))
        
    print(classification_report(y_test, cnv_preds))
    counter += 1
    
    
model_name = "FOLD_dnn_clf.model"
model_path = "models/" + model_name
model.save(model_path)
