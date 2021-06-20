# -*- coding: utf-8 -*-
'''Prediction Using Decision Tree Algorithm'''
#importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

#Label Encoding of Categorical data in y
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Building the Classifier model
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtc.fit(X_train, y_train)

#Predicting the test values
prediction = dtc.predict(X_test)

#Confusion Matrix and Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, prediction)
acc = accuracy_score(y_test, prediction)

#Visualizing the Training data
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
classifier = dtc
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 3].min() - 1, stop = X_set[:, 3].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel(), X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(0, X1.max())
plt.ylim(0, X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 2], X_set[y_set == j, 3],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
classifier = dtc
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 3].min() - 1, stop = X_set[:, 3].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel(), X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(0, X1.max())
plt.ylim(0, X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 2], X_set[y_set == j, 3],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()

'''Conclusion - only 1 prediction is incorrect'''


