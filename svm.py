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
labels = preprocessing.scale(labels)
features = features.drop('GHI',axis= 1) #now remove label from features
#axis = 1 is a reference to the columns
features = features.drop('DHI',axis= 1)
features = features.drop('DNI',axis= 1)
#features = features.drop('DHI_1',axis= 1)
#features = features.drop('DNI_1',axis= 1)
#features = features.drop('GHI_1',axis= 1)
#features = features.drop('Snow Depth',axis= 1)
features = features.drop('Year',axis= 1)
print ('after dropping:', features.head(5))
feature_list = list(features.columns) #to save the feature names into a list
features = np.array(features) #convert to a nuumpy array
features = preprocessing.scale(features)

from sklearn.model_selection import train_test_split
#x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#example on how to use trainn_test_split to split the data into test and training
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
#to confirm that our data is same/safe
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#baseline error (dont know)
# The baseline predictions are the historical averages
#baseline is the error we would get if we simply predicted the average max temperature for all days.
#baseline_preds = test_features[:, feature_list.index('GHI')]
# Baseline errors, and display average baseline error
#baseline_errors = abs(baseline_preds - test_labels)
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

#TRAINING
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
#rf = RandomForestRegressor(n_estimators = 102, random_state = 42)
# Train the model on training data
#rf.fit(train_features, train_labels)

from sklearn.svm import SVR
SVR_model = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
SVR_model.fit(train_features,train_labels)
predictions = SVR_model.predict(test_features)
print('SVM_ SCORE',SVR_model.score(test_features,test_labels))
#TESTING
# Use the forest's predict method on the test data
#predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
#print('Mean Absolute Percentage Error: ',np.mean(mape))
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(test_labels, predictions))
print('rms:',rms)

# Get numerical feature importances
#importances = list(rf.feature_importances_)
# List of tuples with variable and importance
#feature_importances = [(feature, importance) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
#feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
#for pair in feature_importances:
#	print('Variable: {:20} Importance: {}'.format(*pair))