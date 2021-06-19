# -*- coding: utf-8 -*-

#Solving the problem using K-means Clustering

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
lc_y = LabelEncoder()
y_transformed = lc_y.fit_transform(y)

#ELBOW Method to find Optimal Number if Clusters
from sklearn.cluster import KMeans
elbow_inertia = []
for i in range(1, 16):
    km = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    km.fit(X)
    elbow_inertia.append(km.inertia_)
plt.plot(range(1, 16), elbow_inertia)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#Since we have the graph almost constant after n=3 clusters, so optimal number of clusters is 3

#Fitting the Model
kmeans_model = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
prediction = kmeans_model.fit_predict(X)

# Visualising the clusters
plt.scatter(X[prediction == 1, 2], X[prediction == 1, 3], s = 50, c = 'red', label = 'Iris-setosa')
plt.scatter(X[prediction == 2, 2], X[prediction == 2, 3], s = 50, c = 'blue', label = 'Iris-versicolor')
plt.scatter(X[prediction == 0, 2], X[prediction == 0, 3], s = 50, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans_model.cluster_centers_[:, 2], kmeans_model.cluster_centers_[:, 3], s = 50, c = 'yellow', label = 'Centroids')
plt.title('Clustering of Species')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()

#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_transformed, prediction)

#In y_transform it is in order [0, 1, 2], but in prediction it is in [1, 2, 0]
'''Hence 0 is 1, 1 is 2, and 2 is 0'''
'''ACCURACY CALCULATION'''
acc = (50+48+36)/150

'''THANK YOU!'''



