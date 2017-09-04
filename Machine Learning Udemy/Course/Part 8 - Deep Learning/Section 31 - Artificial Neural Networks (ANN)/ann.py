# Artificial Neural Network

# Installing Theano
# this is an open source numerical computation lib for fast computation
# it can run both on CPU and GPU
# Theano makes use of the GPU since it is much more powerful and can run much more float point calc than CPU
# GPU is more specialized for extensive highy computation taks and parallel computations
#  we forward propagate the activation of neurons, it needs parallel computations
# same with back propagations
# developed by machine learning experts at Uni of Montreal
# run the below command in a terminal or cmd
# conda install theano pygpu -> use this since the below command is causing Github error

#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# also runs on GPU, used in deep learning field
# both Theano and Tensorflow are mainly for research, to implement a NN you should do it from scratch
# that is why we use Keras which makes it easy to create NN, created by a Google employee
# just like we use Scikit to use machine learning models, we use Keras for deep learning
# while setting up tensorflow, it is better to choose installation with GPU
# can get one for cheap from AWS
# for the dataset we used, we can go with CPU

# conda create -n tensorflow python=3.5
# pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.1-cp35-cp35m-win_amd64.whl
# above command 1st caused an error, it needs python 3.5
# so used this first: conda install python=3.5
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# we dont need the first 3 columns as they are irrelevant to the output
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# encoding has to be done prior to data split into train and test
# encoding country column and gender column
# since our categorical colums are not ordinal, no relational order, so we need dummy variables
# we are creating dummy variables for column1 because it has 3 categories, we remove one and it will have 2
# we are not doing it for column for gender because it is binary, if we remove one it will be left with one, which makes no difference
# we need to remove one column of dummy variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# model_selection is same as cross validation, its just an updated method of the latter

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
# sequential is used to initialize the neural network
# Dense is used to build the layers of ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# 11 is for number of independent variables we have in dataset
# we will use rectifier function for hidden layer and sigmoid for output layer
# dim is number of nodes we are adding in hidden layer
# there is no rule as to how many nodes we should use
# choose nodes as the average of number of nodes in input layer and output layer
# if you want be artistic, do parameter tuning like k fold cross validation
# in CV you have set besides train and test set, and experiment with different nodes on that set
# we chose 6 here because average value of 11 and 1 is 6
# init is used to initialize the weights to small numbers close to 0
# activation is AF we choose, in hidden layer we use rectifier, relu
# we need to mention input_dim only for the first hidden layer
# because since its teh first it doesnt know what the input is going to be
# for subsequent hidden layers, we dont have to specify the nodes

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
# we dont need dim here
# 2nd hidden layer knows what to expect

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
# we want only one node in output layer, so output_dim = 1
# from second hidden layer to putput layer we need weights too, so we have init
# we are using sigmoid because it works best with classification using probability
# if our classification is not binary, then we should set dim to number of categories
# sigmoid has to be changed to softsigmoid
# softsigmoid works with dependent variable that has multiple classes

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# optimizer is the algo you want to use to set the optimal set of weights in the NN
# the algo is Stochastic gradient descent, there are different types, the best one is called adam
# loss function helps in optimising weights, similar to sum of squares
# since we are doing a binary classification, we used binary_crossentropy
# for multiple classification, we use categorical_entropy
# metrics is used to evaluate the model
# when we fit our model, we see that our accuracy gradually increases

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# nb_epoc is number of rounds NN should go through all the data
# there is no rule of thumb for batch or epoch
# we are going with values 10 and 100
# 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
# y_pred contains probabilities
# for calculation of CM ,we need booleans
# so we convert y_pred into 1s and 0s
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
# using the predicitons made, the bank can target customers who might leave
# they can then study more features of those leaving customers to know whats going wrong

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)