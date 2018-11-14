# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:59:34 2018

@author: austi
"""
import numpy as np

from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
scaler = StandardScaler()
scaler.fit(iris.data)
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree._tree import TREE_LEAF
import graphviz 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def printTable(data):
   with pd.option_context('display.max_rows', None, 'display.max_columns', None):
       print(data)

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
    X_train, X_test, Y_train, Y_test = train_test_split(carData, carTarget, test_size=0.33, random_state = 42)
        
    #Train model with the decision tree
    classifier = tree.DecisionTreeClassifier(max_depth=100)
    #classifier = classifier.fit(iris.data, iris.target)

    classifier = classifier.fit(X_train, Y_train)
    features = ['Buying','Maintence','Doors','Persons','Lug_boot','Saftey']
    classes =  ['unacc','acc','good','vgood']
    dot_data = tree.export_graphviz(classifier, out_file=None, max_depth =  4, impurity = False, proportion = True, feature_names=features, class_names=classes, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('test-output/car')
    
    predictions = classifier.predict(X_test)
    print("Car dataset: Accuracy is ", round(accuracy_score(Y_test,predictions)*100),'%')
    




def overwatch():
    column_names = ['Team Stack','Role','Leaver','Mode','Map','Result']
    features =  ['Team Stack','Role','Leaver']
    classes = ['Loss','Win']
    data = pd.read_csv("./overwatch/Overwatch.csv",names = column_names)
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
    data['Team Stack'][data['Team Stack'] == '1']= 0
    data['Team Stack'][data['Team Stack'] == '2']= 1
    data['Team Stack'][data['Team Stack'] == '3']= 1
    data['Team Stack'][data['Team Stack'] == '4']= 1
    data['Team Stack'][data['Team Stack'] == '5']= 1
    
    #Make Leaver columnn binary (if its on enemy team, doesnt affect our team really)
    #This will make it binary once everything goes numeric
    data['Leaver'][data['Leaver'] == 'Enemy team'] ='No'
    
    #Make Match result binary (Draw = lose)
    data['Result'][data['Result'] == 'Draw'] ='Loss'   

    
   #TURN THIS INTO ONEHOTENCODER BECAUSE WE DONT WANT IT NUMERIC
   
    #turn the categorized data into numerical
    
    LEncoder = LabelEncoder()
    data = data.apply(LEncoder.fit_transform)
    
    #split arrays into values and targets (classif)
    overwatchData = data.iloc[:,[0,1,2]]
    overwatchTarget = data.iloc[:,[3]]
    
    
    #max_depth restricts model so it doesn't grow complex and overfit - similar to pruning
    X_train, X_test, Y_train, Y_test = train_test_split(overwatchData, overwatchTarget, test_size=0.33, random_state = 42)
    
    
    #Train model with the decision tree
    classifier = tree.DecisionTreeClassifier(max_depth=5)
    #classifier = classifier.fit(iris.data, iris.target)

    classifier = classifier.fit(X_train, Y_train)


    dot_data = tree.export_graphviz(classifier, out_file=None, max_depth =  4, impurity = False, proportion = True, feature_names=features, class_names=classes, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('test-output/overwatch')
    
    predictions = classifier.predict(X_test)
    print("Overwatch: Accuracy is ", round(accuracy_score(Y_test,predictions)*100),'%')
   


def main():
    seed = 42
    
    #lets try to bin the data
#    bins = np.linspace(0, 10, 5, retstep=True)
#    print(bins)
#    data = np.random.random(100)
#    print(iris.data)
#    print(data)
#    iris.data = np.digitize(iris.data, bins)

    #Lets try again! bin th edata by converting to pandas, binning, then converting back to numpy
#    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                     columns= iris['feature_names'] + ['target'])
#
#    # Bin the data frame by "a" with 10 bins...
#    bins = np.linspace(df['sepal length (cm)'].min(), df['sepal length (cm)'].max(), 10)
#    groups = df.groupby(np.digitize(df['sepal length (cm)'], bins))
#    
#   
#    # Apply some arbitrary function to aggregate binned data
#    groups = groups.aggregate(lambda x: np.mean(x[x > 0.5]))
#    print(groups)
#    groups = groups.values
    
    
    #NOTE: THE ABOVE DIDNT WORK! BELOW IS MY SECOND ATTEMPT AT BINNING, WITH ONLY 2 BINS
    
#    bin1 = np.empty([150, 4])
#    bin2 = np.empty([150, 4])
#    bin1count = 0
#    bin2count = 0
#    for i in range(0,149 ):
#        #Sepal length (cm)
#        if iris.data[i][0] < 5.5 :
#            bin1[bin1count] = iris.data[i]
#            bin1count = bin1count + 1
#        else:
#            bin2[bin2count] = iris.data[i]
#            bin2count = bin2count + 1i
        

    #max_depth restricts model so it doesn't grow complex and overfit - similar to pruning
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state = seed)
    
    
    #Train model with the decision tree
    classifier = tree.DecisionTreeClassifier(max_depth=5)
    #classifier = classifier.fit(iris.data, iris.target)
    classifier = classifier.fit(X_train, Y_train)

    
    dot_data = tree.export_graphviz(classifier, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('test-output/iris')
    
    predictions = classifier.predict(X_test)
    print("Iris: Accuracy is ", round(accuracy_score(Y_test,predictions)*100),'%')
   
    #Additional data sets:
    #https://www.kaggle.com/spacex/spacex-missions
    #https://www.kaggle.com/edopic/overwatch

    
    overwatch()
    
    df = pd.read_csv("car/car.csv")
    UCI_carEval(df)
    
if __name__== "__main__":
    main()
 
