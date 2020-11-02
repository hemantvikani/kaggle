# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:56:35 2020

@author: heman
"""
import numpy as np
import warnings
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style

def k_nearest_neighbors(data,predict,k=3):
    if (len(data)) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance  = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
result = k_nearest_neighbors(dataset, new_features)
print(result)

for i in dataset :
    
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i)
        
plt.scatter(new_features[0], new_features[1], s=100)


plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.show()