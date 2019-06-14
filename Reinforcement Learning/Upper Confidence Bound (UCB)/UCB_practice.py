# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:26:19 2018

@author: Akhilesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset : this is what god knows the user is gonna do
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

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
        if(number_of_selections[i] > 0):  #perform this after after at least one selection of each ad
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])    
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1 
    reward = dataset.values[n, ad]     
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

#visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ad selections')
plt.xlabel('Ads')
plt.ylabel('Number of times the ad was selected')
plt.show()   



        