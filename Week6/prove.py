# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:17:33 2018

@author: austi
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("car.csv")

#turn the categorized data into numerical
for col in data.columns:
    data[col] = pd.Categorical(data[col], categories=data[col].unique()).codes


#Handle missing data:
#There is none!

#Fit data?
#scaler = StandardScaler()
#scaler.fit(iris.data)


#Split data into test and training sets here
#dataframe.iloc : gets columns based on indicies
carData = data.iloc[:,[0,1,2,3,4,5]]
carTarget = data.iloc[:,[6]]
X_train, X_test, Y_train, Y_test = train_test_split(carData, carTarget, test_size=0.33, random_state = 40)

#Train model with the NeuralNet:

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1, activation='relu')

clf.fit(X_train, np.ravel(Y_train))

y_predict = clf.predict(X_test)
print("Car dataset: Accuracy is ", round(accuracy_score(Y_test,y_predict)*100),'%')
    
    





