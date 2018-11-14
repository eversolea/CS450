# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:59:34 2018

@author: austi
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
scaler = StandardScaler()
scaler.fit(iris.data)
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF
import graphviz 
import pandas as pd




def main():
    seed = 42
    #iris.data = scaler.transform(iris.data)
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state = seed)




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

    print ("binning completed")

    #max_depth restricts model so it doesn't grow complex and overfit - similar to pruning
    classifier = tree.DecisionTreeClassifier(max_depth=5)

    classifier = classifier.fit(iris.data, iris.target)
    
     
    
    
    
    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render()
    graph.render('test-output/testMax_depth')
    
    classifier = tree.DecisionTreeClassifier()

    classifier = classifier.fit(iris.data, iris.target)
    
    
    
    #post-pruning
    #Source: https://stackoverflow.com/questions/49428469/pruning-decision-trees
    def prune_index(inner_tree, index, threshold):
        if inner_tree.value[index].min() < threshold:
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are shildren, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            prune_index(inner_tree, inner_tree.children_left[index], threshold)
            prune_index(inner_tree, inner_tree.children_right[index], threshold)

    print(sum(classifier.tree_.children_left < 0))
    # start pruning from the root
    prune_index(classifier.tree_, 0, 5)
    sum(classifier.tree_.children_left < 0)
    
    
    
    
    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render()
    graph.render('test-output/testPruning')
    
    #Additional data sets:
    #https://www.kaggle.com/spacex/spacex-missions
    #https://www.kaggle.com/edopic/overwatch
    
    
    
if __name__== "__main__":
    main()
 