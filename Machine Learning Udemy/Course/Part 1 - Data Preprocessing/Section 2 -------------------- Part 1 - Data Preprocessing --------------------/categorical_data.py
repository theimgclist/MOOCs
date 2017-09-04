# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
# Lecture 12
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# the below line encodes the first or country column to numberical values
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #giving first column of all rows
# categorical_features parameter specifies which column we are using
onehotencoder = OneHotEncoder(categorical_features = [0])
# X column gets replaced by 3 dummy categorical columns

X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
# since we used labelencode_X for fitting X variable, we need a new one for y
# since it is a dependent variable, the LabelEncoder knows its a category and hence doesnt put any ordering into it
# so we dont need OneHotEncoder here
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)