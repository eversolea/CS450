# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()

#Make my custom Alogrithm class
class HardCodedClassifier:
    
    def fit(self, data_train, targets_train):
        m = HardCodedModel()
        return m

class HardCodedModel:
    
    def predict(self, data_test):
        perdiction = np.zeros(len(data_test))
        for i in range(len(data_test)):
            #I would do my custom algorithm here
            perdiction[i] = 0
        return perdiction
            
#Added a main! sweet!
def main():
    # Show the data (the attributes of ech instance)
    print(iris.data)
    
    # Show the target values (in numeric format) of each instance
    print(iris.target)
    
    # Show the actual target names that correspond to each number
    print(iris.target_names)
    
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state = 42)
    
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    print(clf)
    perdiction = clf.predict(X_test)
    print(Y_test)
    print(perdiction)
    
    counter = 0
    for i in range(len(perdiction)):
        if perdiction[i] == Y_test[i]:
            counter = counter + 1
    
    print("Accuarcy: " + str(counter/len(perdiction)))
    
    classifier = HardCodedClassifier()
    model = classifier.fit(X_train, Y_train)
    perdiction = model.predict(X_test)
    
    counter = 0
    for i in range(len(perdiction)):
        if perdiction[i] == Y_test[i]:
            counter = counter + 1
    
    
    print(Y_test)
    print(perdiction)
    
    print("Accuarcy: " + str(counter/len(perdiction)))

if __name__== "__main__":
  main()