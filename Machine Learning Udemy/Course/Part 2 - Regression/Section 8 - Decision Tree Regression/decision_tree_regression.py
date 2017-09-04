# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
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

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
# with polynomial regression and SVR
# we had predicted value which was close to actual salary
# but with decision tree, we get exact 150000
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results (higher resolution)
# in our before examples, we were using grid for smoother and higher resolution
# if we didnt use grid, then the curve we got would be just like what we had with before regression models
# but it shouldnt be, because the models before decison trees were continuous
# so its ok to have a curve that joined 2 data points with a curve
# but decision trees are non continuous, there will be splits
# so we should use grid here to see the difference
# we get the shape of stairs
# DTs are not effective when used with 1D
# if we add more dimensions, then the model with DT will give good performance



X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()