# Natural Language Processing

# Importing the dataset
# we are using delim because its default separator is tab
# we can also use read.csv with delimiter as tab
# quote is to ignore quotes in the review
# we ignore quotes because we dont usually need them from the text
# 
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# we are converting words into lowercase so that we dont have same word in different case as different features in sparse matrix

install.packages('tm')
install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
# enter dtm in console to see whats the % of sparsity
# after removingsparseterms, columns changed from 1500 to 600, sparsity decreased to 99%
dtm = DocumentTermMatrix(corpus) # creates sparse matrix of features or words
dtm = removeSparseTerms(dtm, 0.999) # keep 99% of most frequent words
dataset = as.data.frame(as.matrix(dtm)) # our classification models need a dataset
dataset$Liked = dataset_original$Liked

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
# we got 74% accuracy rate
cm = table(test_set[, 692], y_pred)