# Eclat
# this is simple and a simplified version of Apriori
# Here its not like people who bought this also bought that
# its more like, people who frequently bought x and y together,
# then others buying x will also buy y
# there is no confidence and support need not be calculated

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv')
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
# we start with support = 0.003
# since Eclat returns most frequently bought products together
# we should give atleast 2 products, so minlen = 2
# we have 845 sets here instead of rules we had in Apriori


rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

# Visualising the results
# since there is no lift, we are using support for sorting the transactions
# if you just want products which are bought together, go with Eclat
# it can be used with simple information
# for larger data and wider functionaity, go with Apriori

inspect(sort(rules, by = 'support')[1:10])