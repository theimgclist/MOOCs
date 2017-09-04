# Natural Language Processing
# NLP is about analysing texts, these can be books,HTML web pages and all sorts of text
# branch of ML
# We are using review classifiers in this example
# We can use the same for genre classification, newspaper article categorisation
# csv and tsv are two file types
# in csv, columns are space separated
# we have two columns, one is review and the other is liked
# 1st review contains review, second contains 0 or 1
# 1 for positive review, 0 is for negative review
# in tsv, columns or values are separated by a tab
# which one is better?
# tsv is better. Because, our reviews might contain comas, which is the same as delimiter in csv
# but tab used as delimiter in tsv is not used in a review, because clicking on tab goes to next button or option
# if you still want csv, we can add something to separate review from liked values
# we have 1000 reviews in the dataset
# for importing tsv, we use the same method read.csv with a different parameter
# we should specify the demiliter here to tab '\t'
# because for read_csv the default delimiter is coma
# quoting = 3 is to ignore the double quotes.
# its good to ignore quotes in NLP


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# this is the first step in data preprocessing in NLP
# instead of reviews, we can have written speeches or articels and we can come up with a prediction or classification
# we use bag of words model
# it will only take the relevant words from each sample or review
# and use only those words for classifying
# it will ignore things like punctuation
# different stem words are taken as one word
# loved, loving are taken as love
# it then uses tokenisation which splits all reviews into tokens or words
# and then attribute one column for each word and count the frequency of each word against each review
# there will be a lot of 0s since for each review there will be many words which dont exist in that review
# this has to be handled and will be taken care of as the last step

# we first take 1st review, and apply cleaning techniques
# then we apply the same on the rest of the reviews
# re is the library that has good tools to clean text
# dataset['Review'][0] returns 1st review
# re.sub - only keep the letters, remove numbers, punctuations, symbols
# 1st parameter to sub is what we dont want to remove
# we use Regular expressions to specify what we want and what to ignore
# the second parameter we used as space is to avoid unnecessary words
# when we remove some characters, the rest of them might form a new word
# so to avoid that we are replacing it with space
# in next step, we are converting all characters into lower case
# in third step we split and then remove unwanted words like prepositions
# words like this make no difference whther they are followed by the word love or hate
# PorterStemmer is a very famour NLTK lib
# in order to remove unwanted words, we should first specify what they are
# the stopwords from nltk contains the list of all such irrelevant words
# what we do is, for every word in review we see if its there in stopwords list
# if yes, we remove it from review
# took around 3 mins to download stopwords
# stopwords is a list of words in different languages, so we specify english in for loop
# we downloaded stopwords, but to use it we should import it
# we used set for reading stopwords because set works faster than a list(makes a lot of difference when text is huge)
# next step is stemming: taking the root of a word
# take the word loved from review, and use only the root word of it that is love
# this is to reduce sparsity of word matrix
# we are applying stem for each word we got from before step, ie after refining and removing unwanted words
# once stemming is done, we then join each list item of review and join them to convert review back to string from list
# we are joining them and adding a space in between them for separation
# corpus is a collection of text




import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split() # splits review into list of strings, before review was a string
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
# why we need bag of words model?
# as first step, we cleaned and created a corpus of all the reviews
# in this step, we take all the words in reviews from corpus without duplicates and attribute them to columns
# for each othe word, we will have a column
# we will get a sparse matrix, we try to reduce sparsity
# bag of words helps us to reduce unneeded words
# why we need this model?
# ML model should be trained with some data to classify a review as positive or negative
# in this case, it needs corelation between words in review and outcome/classifier
# this is similar to what we did in classification models like logistic regression, trees etc
# here words we chose for columns are our independent variables
# we should minimize them for better performance
# check the parameters of CountVectorizer, it can do all that we did using for loop by using the parameters
# the manual work can be automated using parameters
# but its not the best way, since doing it manually will teach how it works
# also, manually doing it will give more options
# suppose we use html pages data, its better we manually clean the data 
# X will be our independent matrix or feature matrix
# y contains the output or the classification
# toarray() will convert X to a matrix
# max_features will try to reduce the number of words
# before using that, we have 1565 columns, after using it, got reduced to 1500
# we can test each of the classification model and see which is the best
# but in common, for NLP Naive Bayes or DTs are used
# copy paste the NB model coded earlier, NO CHANGES are needed
# WE dont need scaling because most of the data is 0 or 1. already scaled


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values # dependent variable vector

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# we got about 73% accuracy.
# we had only 1000 observations, if we had more, accuracy would have been more
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)