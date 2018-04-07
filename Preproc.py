import pandas as pd
features = pd.read_csv('temp.csv') #reads the file and saves it into features
print features.head(5) #initial 5 rows
print features.shape #shape
print features.describe()
#features= pd.get_dummies(features) #one-hot encodes data
#print features.iloc[:,5:].head(5) #print first 5 rows and some colums
#print features.shape #shape has increased after one-hot encoding, obviously!

import numpy as np #now we convert panda dataframe to numpy arrays
from sklearn import preprocessing
labels = np.array(features['GHI']) #what we want to predict, the actua values of max temp
print labels
labels = preprocessing.scale(labels)
print labels
features = features.drop('GHI',axis= 1) #now remove label from features
#axis = 1 is a reference to the columns
features = features.drop('DHI',axis= 1)
features = features.drop('DNI',axis= 1)
#features = features.drop('GHI_1',axis= 1)
features = features.drop('Snow Depth',axis= 1)
features = features.drop('Year',axis= 1)
print ('after dropping:', features.head(5))
feature_list = list(features.columns) #to save the feature names into a list
features = np.array(features) #convert to a nuumpy array
print features

features = preprocessing.scale(features)
print features