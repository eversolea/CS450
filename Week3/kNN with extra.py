# -*- coding: utf-8 -*-
import numpy as np
from statistics import mode
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
iris = datasets.load_iris()
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from scipy.io import arff        
#Added a main! sweet!
def main():
    # Show the data (the attributes of ech instance)
    #print(iris.data)
    # Show the target values (in numeric format) of each instance
    #print(iris.target)
    # Show the actual target names that correspond to each number
    #print(iris.target_names)
    #Get training and testing lists
    seed = 42
    
    #Handle numeric data on different scales using the Scaler
    scaler = StandardScaler()
    scaler.fit(iris.data)
    iris.data = scaler.transform(iris.data)
    #X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state = seed)
    
    
    print("Car Evaluation Dataset with kNN Classifier:")
    #load data from car
    def UCI_carEval(data):
        
        #turn the categorized data into numerical
        for col in data.columns:
            data[col] = pd.Categorical(data[col], categories=data[col].unique()).codes
    
        #print(data)
        #print(pd.Categorical(data[col], categories=data[col].unique()).codes)
        
        
        #Handle missing data:
        #There is none!
    
    
        #Split data into test and training sets here
        #dataframe.iloc : gets columns based on indicies
        carData = data.iloc[:,[0,1,2,3,4,5]]
        carTarget = data.iloc[:,[6]]
        X_train, X_test, Y_train, Y_test = train_test_split(carData, carTarget, test_size=0.33, random_state = seed)
        
        
        #Built-in k Nearest Neighbors Classifier: k=3
        print("\nBuilt-in k Nearest Neighbors Classifier: k=3\n")
        neigh = KNeighborsClassifier(n_neighbors=3)
        
        #K cross-fold validation with 10 sections:
        print("K-fold Cross Validation:")
        scores = cross_val_score(neigh, carData, carTarget, cv=10, scoring='f1_macro')
        #According to the SKLearn documentation, If CV is an integer then k-fold cross validation
        #will be used to find the score.
        #http://scikit-learn.org/stable/modules/cross_validation.html
        
        print("Scores:")
        print(scores)
        print("\n")
        neigh.fit(X_train, Y_train)
        perdiction = neigh.predict(X_test)
        print("Original:")
        original = Y_test.as_matrix().tolist()
        print(*original, sep=",")
        print("Perdiction")
        print(perdiction)
          
        counter = 0
        for i in range(len(perdiction)):
            if perdiction[i] == Y_test.as_matrix()[i]:
                counter = counter + 1
        print("Accuarcy: " + str(counter/len(perdiction)))  

    
        X_train, X_test, Y_train, Y_test = train_test_split(carData, carTarget, test_size=0.33, random_state = 11)
         
        #Built-in k Nearest Neighbors Classifier: k=3 with different data set
        print("\nBuilt-in k Nearest Neighbors Classifier: k=3 with different data set\n")
        neigh = KNeighborsClassifier(n_neighbors=3)
        
        #K cross-fold validation with 10 sections:
        print("K-fold Cross Validation:")
        scores = cross_val_score(neigh, iris.data, iris.target, cv=10, scoring='f1_macro')
        print(scores)
        #According to the SKLearn documentation, If CV is an integer then k-fold cross validation
        #will be used to find the score.
        #http://scikit-learn.org/stable/modules/cross_validation.html
        
        print("Scores:")
        print("\n")
        neigh.fit(X_train, Y_train)
        perdiction = neigh.predict(X_test)
        print("Original:")
        original = Y_test.as_matrix().tolist()
        print(*original, sep=",")
        print("Perdiction")
        print(perdiction)
    
         
        counter = 0
        for i in range(len(perdiction)):
            if perdiction[i] == Y_test.as_matrix()[i]:
                counter = counter + 1
        print("Accuarcy: " + str(counter/len(perdiction)))  



        print("\nk Nearest Neighbors Classifier: k=12\n")
        neigh = KNeighborsClassifier(n_neighbors=12)
        
        #K cross-fold validation with 10 sections:
        print("K-fold Cross Validation:")
        scores = cross_val_score(neigh, iris.data, iris.target, cv=10, scoring='f1_macro')
        
        #According to the SKLearn documentation, If CV is an integer then k-fold cross validation
        #will be used to find the score.
        #http://scikit-learn.org/stable/modules/cross_validation.html
        
        print("Scores:")
        print(scores)
        print("\n")
        neigh.fit(X_train, Y_train)
        perdiction = neigh.predict(X_test)
        print("Original:")
        original = Y_test.as_matrix().tolist()
        print(*original, sep=",")
        print("Perdiction")
        print(perdiction)
    
         
        counter = 0
        for i in range(len(perdiction)):
            if perdiction[i] == Y_test.as_matrix()[i]:
                counter = counter + 1
        print("Accuarcy: " + str(counter/len(perdiction)))  
    
         
        print("\nk Nearest Neighbors Classifier: k=30\n")
        neigh = KNeighborsClassifier(n_neighbors=30)
        
        #K cross-fold validation with 10 sections:
        print("K-fold Cross Validation:")
        scores = cross_val_score(neigh, iris.data, iris.target, cv=10, scoring='f1_macro')
        #According to the SKLearn documentation, If CV is an integer then k-fold cross validation
        #will be used to find the score.
        #http://scikit-learn.org/stable/modules/cross_validation.html
        
        print("Scores:")
        print(scores)
        print("\n")
        neigh.fit(X_train, Y_train)
        perdiction = neigh.predict(X_test)
        print("Original:")
        original = Y_test.as_matrix().tolist()
        print(*original, sep=",")
        print("Perdiction")
        print(perdiction)
        
        
        counter = 0
        for i in range(len(perdiction)):
            if perdiction[i] == Y_test.as_matrix()[i]:
                counter = counter + 1
        print("Accuarcy: " + str(counter/len(perdiction)))  
   
    df = pd.read_csv("./car.csv")
    UCI_carEval(df)
    
    
    
    
    print("\n\n\n\nAutism Data)
    #load data from autism data set
    def AutismData(data):
        
        #Handle missing data:
        
        data = data.replace('?', np.NaN)
        
        
        #turn the categorized data into numerical
        data = data.apply(LabelEncoder().fit_transform)
        #Algorithm found at:
        #   https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
        #print(data)
        
        #This contains multiple arrays! Very confusing to unpack.
        
        #Split data into test and training sets here:
        X_train, X_test, Y_train, Y_test = train_test_split(data, test_size=0.33, random_state = seed)
    
        #Built-in Nearest Neighbors class
    
        print("\nBuilt-in k Nearest Neighbors Classifier:\n")
        neigh = KNeighborsClassifier(n_neighbors=3)
        
        #K cross-fold validation with 10 sections:
        print("K-fold Cross Validation:")
        scores = cross_val_score(neigh, iris.data, iris.target, cv=10, scoring='f1_macro')
        #According to the SKLearn documentation, If CV is an integer then k-fold cross validation
        #will be used to find the score.
        #http://scikit-learn.org/stable/modules/cross_validation.html
        
        print("Scores:")
        print("\n")
        neigh.fit(X_train, Y_train)
        perdiction = neigh.predict(X_test)
        print("Original:")
        print(Y_test)
        print("Perdiction")
        print(perdiction)
    
         
        counter = 0
        for i in range(len(perdiction)):
            if perdiction[i] == Y_test[i]:
                counter = counter + 1
        print("Accuarcy: " + str(counter/len(perdiction)))  

    #got this from https://discuss.analyticsvidhya.com/t/loading-arff-type-files-in-python/27419/2
    data = arff.loadarff('autism.arff')
    at = pd.DataFrame(data[0])
    #AutismData(at)

    
   
    
    #CURRENT PROBLEMS:
    
    #AUTISM DATASET WILL NOT SORT
    #AUTOMPG HANDLING MISSING DATA DOES NOT WORK ('[3] is not in index?)
    
    #load data from car
    def AutoMPG(data):
        
        #Handle missing data:
        #The dataset docs said horsepower (col 4) has some missing values
        #Lets replace all columns as a percaution
        data = data.replace('?', np.NaN)
        print(data)
        #turn the categorized data into numerical DOESNT WORK
        for col in data.columns:
            data[col] = pd.Categorical(data[col], categories=data[col].unique()).codes
    
        print(data)
        print(pd.Categorical(data[col], categories=data[col].unique()).codes)
        
        
        
    
    am = pd.read_csv("./auto-mpg.csv")
    #AutoMPG(am)
    
    



if __name__== "__main__":
  main()
 