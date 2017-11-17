# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Start by changing the dataset name
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # all rows, all columns except the last
y = dataset.iloc[:, 1].values # all rows and the second column with index 1
# if you look at the table of variables
# we see that X has two dimensions, rows and columns
# y has only one, because it is saved as a vector
# X is a matrix of features/ independent variables
# y is a vector of dependent variable


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# We will train our model using the train data
# to make predictions using X train to predict y train, to know the corelations
# then the performance of the model is assessed by trying to make predictions on test data


# for some algorithms that we use in Python or R
# the algorithms take care of feature scaling
# SLR does it, so we dont need feature scaling here


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# the above method takes the training data and learns corelation
# SLR is created and fitted to the training data


# Predicting the Test set results
y_pred = regressor.predict(X_test)
# y_pred will be a vector, since it contains dependent variable values
# this will contain all the predicted salaries of our X_test data


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# if we see the plot, we see the difference between actual and predicted data
# for some predictions, actual and predicted values are same, when red dots intersect with blue line
# for some predcitons, it is close
# for few predictions, predicted value is far from actual value
# why are we using predicted values of X_train here?
# try changing regressor.predict(X_train) to y_train
# we will get a criss crossed plot
# regressor gives us a line that best fits that data


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# notice here that we used X_train for plotting
# we trained our model using X_train data
# we used X_train data to fit a model
# so whenever use predict, it uses that already fit model to make predictions
# even if we change X_train to X-test,there wont be a change in line i.e our model
# the line we see is what we got because of regressor.fit()
# like before with train data, we see that the test data too has come up with good predictions
# there will always be a mix of perfect, good, bad, worse predictions
# this is just a simple example: 
# we have only one feature
# we have linear dependency between IV and DV
# this is the first machine learning model
# machine here is our SLR model
# learning means we trained that model with the data available









