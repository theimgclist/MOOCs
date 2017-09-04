# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# we have a dataset of users with some features
# Based on age and salary, we try to predict if user buys a product or not
# so we only take those two features in X
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
# since we are not using linear regression
# we should do scaling for accurate predictions
# we are scaling the train data and test data
# we dont scale the y vector, since its categorical and already scaled


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Logistic Regression to the Training set
# Logistic Regression is a linear model
# that means the two classes of users are goign to be classified using a stgraight line
# use ctrl + I to see what all parameters are available
# we are fitting logisitic regression using the training data
# we train our model on train data
# it finds and learns corelations between X and Y
# which will later help us to predict on test data


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
# we use our trained model to predict values of test data

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# when importing, if its a class, it will be capitalised
# if its a method, it will be in lower case
# first parameter to confusion_matrix is real values or ground truths
# second one is predicted values
# the cm variable is a 2 X 2 matrix
# 65 and 24 denote accurate predictions
# i remember doing a task about Confusion Matrix before
# Must be scikit tutorials from Markham


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
# how and why this part of code is used for plot is not explained
# it will be explained at the end of the section for those who are interested
# on the plot, we see many points, plotted in red and blue
# they are plotted from the training data
# each user has two characteristics
# the first one is age, the second is salary
# for red points, dependent variable purchased value is 0
# for green points, purchased value is 1
# we can deduce that user with lower age, are in red, didnt buy product
# users with higher age in green, bought the product
# also, some users in green, with high age but low salary, also bought the product
# some young people with high salary also bought the product, the SUV
# what are we trying to do?
# the goal is to classify users
# we have two regions divided by a straight line
# the line is called prediction boundary
# it is straight not because it is a random pick
# since log reg is linear, we can only have a straight line
# for non linear regressions, we will have non straight boundary
# our logistic reg model came up with good predictions
# however , some points in green, who are users who bought suv, are in red region
# it is because out model is linear and straight line doesnt cover all linear points
# also because users are not linearly distributed
# the observations we saw are for train data
# using the observations, the company can target users from social network who are of prefered age

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results

# for test data, just like previous models, the model remains the same, its the data that changes
# so for test data, the regions of yes and no will be the same
# it is the data points which change


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# this is how the plotting works
"""
The entire region is taken as pixels.
once our logistic regression comes up with a prediction, that point or pixel is given a color
If its a 0, its given the color red. If its 1, its given a color green
That way, the entire region split by the boundary line is filled with pixels and color
plt.contour() draws the boundary line separating the two regions
"""