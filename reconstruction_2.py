# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 00:00:51 2016

@author: root
"""

import pandas as pd
train=pd.read_csv('train.csv',delimiter =',',skiprows=1)
     
X=train.values[:,1:]
Y=train.values[:,0]

from sklearn.cluster import KMeans
classifier=KMeans(init='k-means++',n_clusters=10)
classifier.fit(X,Y)

test=pd.read_csv('test.csv')
classifier.predict(test)
print classifier.score(X,Y)