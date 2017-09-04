# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# we are appending one column of ones  to X
# it is because the b0 constant value is not considered in the algorithms
# we can assume we have another feature or predictor which we call x0, and we take its value as 1
# that way we not only  have new column of ones
# and also a dummy feature x0
# without the column of ones, the library doesnt take b0 value into account
# we start with having all features in opt array
# we then remove one by one which are not significant
# we cant use the old regressor we got from linear_model
# for backward elimination we need a differenr regressor from a different class which we imported
# ols stands for ordinarily squared
# if you click on help for sm.OLS, we see that first parameter is the dependent variable
# second parameter is the X_opt which contains observations and regressors
# it also mentions that the intercept should be added by the user
# intercept here means the 1s we added
# the summary() method retuns a statistical table

             
# we start by giving all preditors to regressor, then check summary to see which feature has highest p value
# in first iteration, column 2 from X has highest p value
# we then refit regressor using features excluding the one at index 2
# we repeat the same for all features until we have no features whose p values are greater than SL = 0.5
# when we use 0,3,5 we get p value for feature at index 5 to be 0.06
# which is slightly more then SL
# we can include that since it is only slightly greater than Sl
# we can improve performance of model by using different metrics
# finally we have only one independent variable which is R n D which has significant linear coreation with dependent variable
# we can also use other metrics like R squared  Adj R squared for feature reduction

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()