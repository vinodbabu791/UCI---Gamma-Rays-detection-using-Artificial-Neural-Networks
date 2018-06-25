# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 00:25:10 2018

@author: Universe
"""
# import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# import the dataset
magic = pd.read_csv('C:/Users/Universe/Desktop/DataScience/Summer/Data Mining/Project/majic.csv',
                    header=None,
                    names=['fLength','fWidth','fSize','fConc','fConc1','fAysm','fM3Long','fM3Trans','fAlpha','fDist','ray'])

magic.describe()
magic.dtypes
magic.ray.value_counts()

# Encode the output variable as 0 and 1
magic.replace({'ray':{'g':1,'h':0}},inplace=True)
magic.ray.value_counts()

# Shuffle the dataframe rows
magic = magic.sample(frac=1).reset_index(drop=True)

# Separate features and labels
features = magic.iloc[:,0:(magic.shape[1]-1)]
labels = magic.iloc[:,magic.shape[1]-1]

# Scale the features
feature_scale = StandardScaler()
features = feature_scale.fit_transform(features)

# Divide into test and training set
feature_train, features_test,labels_train, labels_test = train_test_split(features,
                                                                          labels,
                                                                          train_size=0.75,
                                                                          random_state=777)

#### Build ANN Model with 2 hidden layers ####

magic_ann = Sequential()

# add layers
# First hidden layer with input layer
magic_ann.add(Dense(10,input_dim=10,activation='relu',init='glorot_uniform'))

# second hidden layer
magic_ann.add(Dense(6,activation='relu',init='glorot_uniform'))

# Third hidden layer
magic_ann.add(Dense(6,activation='relu',init='glorot_uniform' ))

# Output layer
magic_ann.add(Dense(1,init='uniform',activation='sigmoid'))

# adding stochastic gradient descent

magic_ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Train the ANN model
magic_ann.fit(x=feature_train,y=labels_train,epochs=100,batch_size=25)

magic_pred = magic_ann.predict(features_test)
magic_pred = (magic_pred>=0.5).astype(int)
cm= confusion_matrix(labels_test,magic_pred)
