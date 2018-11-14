# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:17:33 2018

@author: austi
"""
import math
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder

def overwatch():
    column_names = ['Team Stack','Role','Leaver','Mode','Map','Result']
    features =  ['Team Stack','Role','Leaver']
    classes = ['Loss','Win']
    data = pd.read_csv("./Overwatch.csv",names = column_names)
    data = data.iloc[1:] #Kill the first row (titles)
    
    
    #Preprocess the rest of the data
    data = data[['Team Stack','Role','Leaver','Result']]

    #Take care of missing data
    #round(data[['Team Stack']].drop([1,2,3,4]).mean(),1)
    #^The result of this is 1.3 -> 1 is the mean
    data[['Team Stack']] = data[['Team Stack']].fillna(1) 
    
    #Filter out only Support and Tank (have to do binary because n-ary won't work with sklearn)
    #df[df.C.str.contains("XYZ") == False]
    

    
    data = data[data['Role'].str.contains("Offense") == False]
    data = data[data['Role'].str.contains("Defense") == False]
    
    
    
    #Change Team Stack to binary (1 ->0, 2-5 ->1)
    data['Team Stack'][data['Team Stack'] == '1']= 1
    data['Team Stack'][data['Team Stack'] == '2']= 2
    data['Team Stack'][data['Team Stack'] == '3']= 3
    data['Team Stack'][data['Team Stack'] == '4']= 4
    data['Team Stack'][data['Team Stack'] == '5']= 5
    
    #Make Leaver columnn binary (if its on enemy team, doesnt affect our team really)
    #This will make it binary once everything goes numeric
    data['Leaver'][data['Leaver'] == 'Enemy team'] ='No'
    
    #Make Match result binary (Draw = lose)
    data['Result'][data['Result'] == 'Draw'] ='Loss'   

   
    #turn the categorized data into numerical
    
    LEncoder = LabelEncoder()
    data = data.apply(LEncoder.fit_transform)
    
    print(data)    
    
    #END PREPROCESSING
    
    #split arrays into values and targets (classif)
    overwatchData = data.iloc[:,[0,1,2]]
    overwatchTarget = data.iloc[:,[3]]
    
    
    
   
    X_train, X_test, Y_train, Y_test = train_test_split(overwatchData, overwatchTarget, test_size=0.33, random_state = 42)
    
    
    #Train model with the Neural network
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), learning_rate='constant', random_state=1, activation='tanh')
    
    clf.fit(X_train, np.ravel(Y_train))
    
    predictions = clf.predict(X_test)
    
    
    print("Overwatch: Accuracy is ", round(accuracy_score(Y_test,predictions)*100),'%')

def cars():
    data = pd.read_csv("car.csv")
    data = data.iloc[1:] #Kill the first row (titles)
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
    

def googlePlay():
    data = pd.read_csv("googleplaystore.csv")
    data = data.iloc[1:] #Kill the first row (titles)
    data['Type'][data['Type'] == 'Free']= 0
    data['Type'][data['Type'] == 'Paid']= 1
    data['Rating'] = data['Rating'].fillna(0)
    data = data.fillna(0)
    
    #appData = np.asarray(data.iloc[:,[2]], dtype="|S6")
    #appTarget = np.asarray(data.iloc[:,[6]], dtype="|S6")
    #appData = data.iloc[:,[0]]
    #appTarget = data.iloc[:,[1]]
    
    #print(appData)
    #print(appTarget)
    print(data['Type'])
    print(data['Type'].shape)
    print(data['Rating'].shape)
    
    X_train, X_test, Y_train, Y_test = train_test_split(data['Type'], data['Rating'], test_size=0.33, random_state = 40)
    
    #Train model with the NeuralNet:
    mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50, max_iter=1000, learning_rate='constant', random_state=42)
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1, activation='relu')
    mlp.fit(X_train.reshape(1,-1), Y_train.reshape(1,-1))
    
    y_predict = mlp.predict(X_test.reshape(1,-1))
    print("Google Play Apps set: Accuracy is ", round(accuracy_score(Y_test,y_predict)*100),'%')
    #ValueError: shapes (3577,1) and (7262,50) not aligned: 1 (dim 1) != 7262 (dim 0)
    
    
def census():
    data = pd.read_csv("UsCensus.csv")

    data = data.fillna(0)
    
    #appData = np.asarray(data.iloc[:,[2]], dtype="|S6")
    #appTarget = np.asarray(data.iloc[:,[6]], dtype="|S6")
    #appData = data.iloc[:,[0]]
    #appTarget = data.iloc[:,[1]]
    
    #print(appData)
    #print(appTarget)
    
    censusData = data.iloc[:,[0,1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
    censusTarget = data.iloc[:,[28]]
    X_train, X_test, Y_train, Y_test = train_test_split(censusData, censusTarget , test_size=0.33, random_state = 40)
    
    #Train model with the NeuralNet:
    mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50, max_iter=1000, learning_rate='constant', random_state=42)
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1, activation='relu')
    mlp.fit(X_train, np.ravel(Y_train))
    
    perdiction = mlp.predict(X_test)
    counter = 0
    for i in range(len(perdiction)):
        if math.isclose(perdiction[i],  Y_test.as_matrix()[i], rel_tol=1e-9, abs_tol=0.0):
            counter = counter + 1
            
    print("Census Dataset Accuarcy: " + str(counter/len(perdiction)))  
    #print(mlp.score(perdiction.reshape(1,-1), Y_test)) This doesnt work yet either cause shapes not aligned
overwatch()
cars()
census() #https://www.kaggle.com/muonneutrino/us-census-demographic-data
#googlePlay() doesn't work right now