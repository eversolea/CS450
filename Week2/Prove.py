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


#Make my custom Alogrithm class
class kNNClassifier:
    
    def fit(self, data_train, targets_train, k):
        m = kNNModel()
        m.init(data_train,targets_train, k)
        return m

class kNNModel:
    
    k = 3 #nearest neighbors
    ranking = np.full((2,k),[[999.0]]) #populate with max number, so gets overwritten quickly.
    input_train = np.zeros(1)
    output_train = np.zeros(1)
    
    def init(self,data_train,targets_train, k):
        self.input_train = data_train
        self.output_train = targets_train
        self.k = k
        
    
    def predict(self, data_test):
        #initialize perdiction, which we will populate with the perdicted values
        perdiction = np.zeros(len(data_test))
        
        for i in range(len(data_test)):
            #reset ranking array every loop
            self.ranking = np.full((2,self.k),[[999.0]])

            #Populate ranking
            for x in range (0,len(self.input_train)):
                distance = self.getDistance(self.input_train[x],data_test[i])
                self.rank(distance,x)

  
            neighbors = np.zeros(self.k,dtype=np.int)# make array to hold training output for the neighbor array
            #Check to see what the majority of the ranking array is
            #Might be able to optimize by trimming 1st Dimension of output_train
            for y in range (0,self.k-1):
                x = int(self.ranking[1,y]) #get ID of element y in 'ranking'
                neighbors[y] = self.output_train[x]
                

            counts = np.bincount(neighbors)
            perdiction[i] = np.argmax(counts) #Set perdiction as the mode of the neighbors training output array
        return perdiction
    
    #The rank function:
    #    distance is the distance from perdiction object to test object.
    #    id is the id of the test object
    def rank(self,distance,id):
        i = self.k-1
        while i >= 0:
            if self.ranking[0,i] > distance:
                self.ranking[0][self.k-1:] = [distance]
                self.ranking[1][self.k-1:] = [id]
                i = -1 #break out of this loop
            i = i - 1 #decrement i each loop
            
        #Here, we will sort the 2Dimensional list by the first row
        #We will use numpy's argsort. Found this usage on:
        #https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        c = self.ranking[0,:].argsort() 
        self.ranking = self.ranking[:,c] 

    #The getDistance function
    #   This gets the distance between the 2 objects in the parameter
    #   Built to handle as many data points as objects contain
    def getDistance(self,trainObject,testObject):
        distance = 0
        for i in range(0,len(trainObject)): #ndim gets the # of dimensions
            distance += (trainObject[i] - testObject[i]) ** 2 #(x-a)^2
        return distance
        
        
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
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state = seed)
    
    #Built-in Guassian class
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
    print("\n\n\n")
    
    #My k-nearest neighbors class 
    print("k-Nearest Neighbor")
    print("k=3:")
    classifier = kNNClassifier()
    model = classifier.fit(X_train, Y_train, 3)
    perdiction = model.predict(X_test)
    
    counter = 0
    for i in range(len(perdiction)):
        if perdiction[i] == Y_test[i]:
            counter = counter + 1
    
    print(Y_test)
    print(perdiction)
    
    print("Accuarcy: " + str(counter/len(perdiction)))


    print("\nk=7:")
    classifier = kNNClassifier()
    model = classifier.fit(X_train, Y_train, 7)
    perdiction = model.predict(X_test)
    
    counter = 0
    for i in range(len(perdiction)):
        if perdiction[i] == Y_test[i]:
            counter = counter + 1
    
    print(Y_test)
    print(perdiction)
    
    print("Accuarcy: " + str(counter/len(perdiction)))

    print("\nk=70:")
    classifier = kNNClassifier()
    model = classifier.fit(X_train, Y_train, 70)
    perdiction = model.predict(X_test)
    
    counter = 0
    for i in range(len(perdiction)):
        if perdiction[i] == Y_test[i]:
            counter = counter + 1
    
    print(Y_test)
    print(perdiction)
    
    print("Accuarcy: " + str(counter/len(perdiction)))

    print("\nk=90:")
    classifier = kNNClassifier()
    model = classifier.fit(X_train, Y_train, 90)
    perdiction = model.predict(X_test)
    
    counter = 0
    for i in range(len(perdiction)):
        if perdiction[i] == Y_test[i]:
            counter = counter + 1
    
    print(Y_test)
    print(perdiction)
    
    print("Accuarcy: " + str(counter/len(perdiction)))

    #Built-in Nearest Neighbors class

    print("\nBuilt-in k Nearest Neighbors Class")
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, Y_train)
    perdiction = neigh.predict(X_test)

    print(Y_test)
    print(perdiction)

     
    counter = 0
    for i in range(len(perdiction)):
        if perdiction[i] == Y_test[i]:
            counter = counter + 1
    print("Accuarcy: " + str(counter/len(perdiction)))  



    print("\n\n\nSome additional playing around:")
    
    n_neighbors = 3
    
    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    
    h = .02  # step size in the mesh
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))
    
    plt.show()

if __name__== "__main__":
  main()