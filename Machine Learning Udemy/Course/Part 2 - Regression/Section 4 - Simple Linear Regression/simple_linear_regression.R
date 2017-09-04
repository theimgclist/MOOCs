# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# dont need this since SLR already takes  care of it
# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# lm stands for linear model
# first argument says salary is directly proportional to yearsexperience
# if we need any info about variables, we can use summaru(regressor) in console to see the result
# if we try that we see some good info
# like the independent variable is given stars
# 0 stars mean no signigicance or not relationship with DV
# 3 stars mean high significance relation with DV
# for salary we got 3 stars, so linear dependency is high between IV and DV
# another thing to see is the p value
# the lower the p value, higher is the significance




# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)




# Visualising the Training set results
install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# in R, we first plot the points, then draw the line and give it a title
# since these are 3 different tasks, we group them using +
# geom_point scatters the points
# geom_line plots the regression line or the model
# aes is the function aesthetic
# for plotting the model, we took x as experience from trainin data
# but y value as predicted salaries
# we dont use actual salaries to fit our model
# because we should only use the predicted salaries with out model
# we cant use y_pred because y_pred is predicted values of test data
# the model will plot the predicted salaries of training data
# the red points denote the actual data
# the blue line denotes the model, the predcited salaries





# Visualising the Test set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')