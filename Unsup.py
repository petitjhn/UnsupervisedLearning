#import dependancies
import pandas as pd
#loading the training data
train=pd.read_csv('train.csv',delimiter =',',skiprows=1)
 #slicing off the labels    
X=train.values[:,1:]
Y=train.values[:,0]
#importing of our classifier
from sklearn.cluster import KMeans
classifier=KMeans(init='k-means++',n_clusters=10)
#training of classifier on dataset
classifier.fit(X,Y)
#testing of classifier on test dataset
test=pd.read_csv('test.csv')
classifier.predict(test)
