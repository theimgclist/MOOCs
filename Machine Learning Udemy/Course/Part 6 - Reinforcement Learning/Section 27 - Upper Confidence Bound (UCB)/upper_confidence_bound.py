# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# ctr is click theough rates
# robots and AI use RL
# in an earlier example, we tried to predict whether a social network user will buy SUV or not
# now the same SUV company has come up with 10 different ads for that SUV
# their goal is to see which is the best ad for using in social media
# see which ad will get the most clicks
# this is a different example from the others done so far
# in the before problems we had data, with some IV and DV
# in this problem we dont really have the data
# for each user that logs in, we show an ad Ad(i)
# if the user clicks it, we set reward to 1, else 0
# depending on the kind of views or clicks the ad got,
# we design our strategy
# the ads are not picked randomly
# suppose for round 10 or user 10, the algorithm uses
# what all it learnt from previous 9 rounds
# and decides which version of the ad to show to the user 10
# that is why RL is also called online/interactive learning
# looking at the dataset, we know that user1 will click if ads 1,5,9 are shown
# this is not known to us before
# so we use UCB and Thompson algos to decide which ad to show for a user u.
# what is instead of any algos, the ads are chosen at random?
# for 10000 trials, the total reward through random ads is around 1200 on an average
# we see that when we used random method to generate which ad to choose, the distribution is uniform
# for UCB, most of the distribution is at 4, whereas with random, it is uniformly distributed across all ads.

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
# since there is no implementation of UBC in Pythong
# it is done from scratch using Python code
''' 
    This algorithm learns from the previous rounds what ad to pick to show to the user.
    For the round n = 0, we wont be having any prior data.
    Since we need some knowledge, we coded it in such a wy that, for the first 10 rounds,
    each round i will be using the ad i
    That way from round 11, we will have the data of previous 10 rounds to work on.
    We coded how the ad gets picked, what about the reward? How do we decide if ad i shown to user u will be clicked?
    For this we use the simulated dataset that contains the data of what ads are clicked and not clicked.
    So the variable reward is being set to the value from dataset
    Since the algorithm uses from previous rounds what ads have better click rate, as our n increses,
    we see that the most frequent ad chosen and clicked from previous rounds will be chosen more often than the rest.
    which is why for the rounds closer to 10000, we see that ads_selected shows more 4s than other numbers.
    So from our dataset, the ad 5(since python is 0 indexed) is the ad with most click rate.
    With random ads, we saw that total reward was around 1200. 
    With this algorithm it is 2200, almost the double. 
    So UCB is much better than random method.
	1e400 is 10 to the power 4

'''
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
# we see that ad 4 or version 5 of the ad is the most clicked ad
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()