# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:49:58 2017

@author: Avinash
"""

# importing the libraries
# a library is a tool used to do a specific job
# give inputs, and it will do the job giving output
import numpy as np # contains mathematical tools to use any math related tasks
import matplotlib.pyplot as plt # used for plotting
import pandas as pd # manage datasets

# importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # take all columns except the last
y = dataset.iloc[:, 3].values
                
# taking care of missing data
from sklearn.preprocessing import Imputer #sklearn is scikit learn with models, imputer class helps us to handle the missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

 
# Splitting the dataset into the Training set and Test set
# test size is 20% of all data
# using random state with a number, will generate same results everytime it is used with same number 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# we use fit_transform for training data
# for test data we just transform, so no fit
# do we have to scale the dummy values?
# the dummy values are already between 0 or 1
# its good if we scale, everything will be on same scale
# but the interpretation is lost
# but the interpretability is lost, we dont know which country looking at the column
# it wont break the model if we dont scale
# even if the ML model we chose doesnt use Eucledian distance, its still good we scale the features
# that way the model will converge better
# eg: Decision trees dont use ED
# but if not scaled, they run for a very long time
# do we have to aplpy scaling for dependent variable
# no, since this is a classification problem, we dont have to
# for regression, the DV scales vastly, then we need scaling
# we dont need to add fit to X_test because for the object sc_X we already fit it using X_train
# for dependent variable y, we will use a new object for standardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)