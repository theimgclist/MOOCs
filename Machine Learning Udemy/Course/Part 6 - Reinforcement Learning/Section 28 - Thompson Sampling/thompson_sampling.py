# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Can thompson do better than UCB?
# Can it do better than UCB in terms of total rewards?
# and will it also give ad version 5 as the best one for using?

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
'''
    For each round or user, we find a random distributed value for each ad version
    That way we find random Distributed values of all ads
    WE take the one the with max value and assign it to the user
    The user gets to see that ad
    Like we did in UCB, the reward is assigned using the simulated dataset
    We see that thompsons algorithm came up with better reward, 2617 vs 2200
    Since we used random distribution, the reward will keep changing
    Thompson also gives ad version 5 as the best ad
'''
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()