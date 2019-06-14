# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 03:03:11 2018

@author: Akhilesh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#found out from th dendrogram that the optimal number of clusters is 5
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 20, c = 'blue', label = 'Careless')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 20, c = 'red', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 20, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 20, c = 'cyan', label = 'Sensible')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 20, c = 'magenta', label = 'Careful')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()