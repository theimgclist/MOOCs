# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
# help the grocery store improve its sales
# associate rule learning can be used to learn where to place different products
# collabortive filtering, accounting based or item based collaborating are other models used for recommender systems
# unlike all previous examples, we dont import the class we use for modeling
# we use the one taken from Python software foundation
# the 1st column also contains data, so we are setting there is no header in th dataset
# the data set contains items bought in 7500 transactions
# apriori is expecting a list of lists as input
# first loop to iterate through each transaction
# inner loop is for each column, there are 20 in total
# see the transactions variable for how dataset is stored as list of lists

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
# here apyopri is the local class available in the working directory
# apriori method takes transaction as input and gives rules as output
# we start with min_support = 0.003 and min_confidence = 0.8
# min length is the min number of items in purchasing basket
# we chose support as 0.003 because we are taking item bought atleast thrice a day
# thats 21 times a week, 21/7500 = 0.003 approximately
# R had a default value for confidence which was 0.8
# confidence of 0.8 means that rule has to be correct at 80% of the time

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
# apriori can be combined with collaborative filtering and user profiles 
# and add additional relevant info and also other more advanced models
# like neighborhood model and latent factor model

results = list(rules)