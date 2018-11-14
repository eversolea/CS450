# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:09:38 2018

@author: austi
"""

from __future__ import print_function

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score


data = pd.read_csv('./overwatch/Overwatch.csv')


#Take care of missing data
data = data.replace('', np.NaN) 
columnMean = round(data[['Team Stack']].mean(),0)
data[['Team Stack']] = data[['Team Stack']].fillna(columnMean)

X = data.iloc[:,[0,1,2,3]]
y = data.iloc[:,[4]]

for clf_type in [RandomForestClassifier, ExtraTreesClassifier]:
    clf = clf_type(n_estimators=100, max_features=None,
                   min_samples_leaf=10, random_state=14823,
                   bootstrap=False, max_depth=None)
    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=174382).split(X, y)

    for train, test in cv:
        clf = clf.fit(X[train], y[train])
        probs = clf.predict_proba(X[test])
        print('{} AUC: {}'.format(clf_type.__name__,
                                  roc_auc_score(y[test], probs[:, 1])))
    print(clf.feature_importances_)