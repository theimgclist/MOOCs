# Apriori
# optimise the sales in a grocery store
# use the association rule learning how and where to place the products
# separate the products which the consumers usually buy together
# that way user goes to al lcorners in store and there is high chance of buying something unplanned
# people who bought this also bought this is an example
# collaborative filtering uses association rule learning

# Data Preprocessing
# in the dataset we see the header column doest contain column names
# that too is a sample of data
# so we are setting Header to false so that it will be taken as the first sample of data
# the dataset contains different transactions done by customers who on average visit store once a week
# each row is a customer who made some purchase of items in the columns
# we dont directly use this dataset for training our model
# it is because the package we use arules doesnt take a csv file as input or dataset
# it takes as input a sparse matrix, matrix with large number of zeros and less non zero values
# what we do is, we tak all the different products
# in this dataset there are around 120 of them
# so we re arrange the data in such a way that, there will be 120 colums
# each column represents a product
# for each customer, the column contains 1 if it is purchased
# 0 otherwise, that is if the product is not purchassed
# for example, a customer who bought only 1 product, will have 1 in the corresponding colum
# 0 in all other columns
# we didnt mention sep as , for read.csv. Because in a csv file, different columns are separated by , by default and read.csv is designed to work with default value
# but transactions needs to know the separator.
# there are some anomalies in the data, like having duplicates.
# so we are removing them
# to train the Apriori algo we shouldnt have any duplicates.
# after running transactions method, we get the result saying there are 5 samples with duplicate data
# in summary we see informative data like num of rows, colums, density which means the proportion of non zero values which in this case is 0.03
# itemFrequencyPlot is a function in arules class
# topN = 10 shows 10 bar plots, top 10, this is useful in knowing which is recommended product

install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# the choice of confidence and support has no general rule, it depends on biz need and model
# in the graph we see that the products on right end are not purchased often
# we should eliminate those from our model
# to do this, we should set an appropriate support
# lets say a product is bought 3 times a day, 21 times in a week
# its support will be 21/7500 which is close to 0.003
# we start confidence with default value(0.8) and then change accordingly
# we started with 0.8, divided it by 2 to try with 0.4 and then with 0.2
# support = 0.03 means product bought atleast thrice everyday
# with 0.03 we had around 1300 rules, with 0.04 we have only 811 rules
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
# we are sorting the rules based on their lift
# people who buy mineral water and wheat pasta, will buy olive oil with 40% chance
# confidence column gives the % of associated product being purchased
# 1:10 says list out first ten samples
inspect(sort(rules, by = 'lift')[1:10])