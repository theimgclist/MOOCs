# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# random forest is a collection of decision trees
# we split dataset into random sets each with k samples
# we predict a y value from each of decision trees
# we average all the y predicted values and take it as the y value
# like decision trees, randon forest regression is also non continuous
# keep viewing the help to know different parameters of library functions


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
# when n_estimators was taken as 10. we got y value as 167000
# when we increased it to 100, we got y = 158000
# when its made e300, we got y value as 160000 which was what the employee stated
# one observation: as we changed n estimator value
# we notice that the number of steps didnt increase or change much
# it is because the y predicted value converges for many trees as the number increases
# polynomial regression predicted the accurate value
# randon forest regression too predicted exactly

y_pred = regressor.predict(6.5)







# Visualising the Random Forest Regression results (higher resolution)
# by using Random forest, we have many decision trees instead of one
# we have more steps in between points instead of one
# this is why we get better predictions
# the intervals between points, which used to be on same level before
# are not split into steps
# it is optimal not to have too many decision trees
# because if we do,then more trees predicted value will converge to some y value



X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()