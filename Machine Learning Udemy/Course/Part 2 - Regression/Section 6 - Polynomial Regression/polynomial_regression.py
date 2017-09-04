# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# In this bluff detecting problem, we try to use the data and predict if an employee is quoting correct salary or bluffing
# in the dataset, we see that there are job positions and level
# normally, we may have to encode Position column since it isnt numeric
# but since we have level column already which does that, we simply ignore Position column from X
# we only need one column for X, but we used 1:2 as index
# it is because, if we directly gave one idex, X will be a vector
# it is always recommended to have X as a matrix
# for that purpose we gave index as 1:2 which again takes only the column at index 1

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
                
                
                
# we dont have to split the data. Reasons being:
# we have just 10 observataions in our dataset, which is less
# if we further split it, our training model may not perform well
# since the employer wants to know if employee is bluffing about salary,
# the value we are trying to predict should be an accurate one
# so the more data we use for regression, the better


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

                                                   
# we dont have to do scaling since it is taken care of by the class                                                 
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

                       
                       
                       
# we are creating a linear regression model just to have it as a reference to compare it with polynomial regression
# we dont have to do this
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)




# we use PolynomialFeatures for handling some polynomial regression tasks
# PolynomialFeatures takes in a feature and creates other features which are simply X feature with different degree or power
# the poly_reg takes X and transforms it into a matrix with each column containing X value raised to some power
# if degree is 2, then X_poly will have 3 columns
# the 1st column will have X to the power 0
# the second column will have X to the power 1
# the third will have X to the power 2
# remember in MLR we added a column of 1s to X for having feature X0
# that is done by default here using poly_reg


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
# on getting the plot, we see that the linear regression model has rather performed poor
# the best fit line gives predicted salaries which are far from actual salaries

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




# Visualising the Polynomial Regression results
# the polynomial regression gives a far better result than that of linear
# we started with degree 2, by changing that to 3 we wil get better results
# by changing it to 4, we get a line that goes through all points
# for the predict function below, we used the polynomial object, since it is required for the polynomial regression
# though we have X_poly variable which is equal to the value passed for predict,
# we have again specified it so that the updated X content will be taken

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# the above plot gives us accurate results
# but the curve draws straight lines between points
# we can change that to curved lines by using X_grid
# here we arrange X values from min to max with increment of 0.1 or 0.01 for even better graph

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))