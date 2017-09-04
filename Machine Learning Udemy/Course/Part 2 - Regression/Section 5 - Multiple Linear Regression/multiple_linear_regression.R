# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set
# unlike SLR, we dont specify features in formula
# sicne we have more than one features, we used a dot .
# profit is a linear combination of independent variables
# we can also yse formula = Profit ~ R.D.Spend + Administraion + market.Share + State
# dots in between feature names denote spaces in feature actual names

regressor = lm(formula = Profit ~ .,
               data = training_set)

# going through the summary, we realise that its R and D which is the most significat feature
# so we can simply use R n D as IV for better performance


# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)




# Building the optimal model using Backward Elimination
# In python we used a different regressor for backward elimination
# in R we will use the same lm regressor
# we dont have dummy variables in R
# since we directly encoded each possible category of State to 3 different numeric values
# we can use both train_data and entire dataset to train and fit the model
# *** is highly significant
# ** is very significant but less than ***
# * good impact
# . - certain level of significance
# summary of 1st iteration shows us that state2 has about 0.99
# we can remove it
# but we also see that state 3 has p value of 0.94
# even if we remove state 2, there is no way state3 will decrease to less than 0.05
# so we can remove them together, which is removing the state feature


regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
# Optional Step: Remove State2 only (as opposed to removing State directly)
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + factor(State, exclude = 2),
#                data = dataset)
# summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)