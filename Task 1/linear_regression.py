# -*- coding: utf-8 -*-
#TASK - 1
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##################

#Importing the Dataset
dataset = pd.read_csv('student_scores.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Dividing into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Building Regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Values
prediction = regressor.predict(X_test)

#Finding the score for 9.25 hrs/day of study
X_given = [[9.25]]
y_tofind = regressor.predict(X_given)

#Graphical Analysis - Training Dataset
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hrs/day vs Score (Training set)')
plt.xlabel('Hrs/day')
plt.ylabel('Score')
plt.show()

#Graphical Analysis - Test Dataset
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hrs/day vs Score (Test set)')
plt.xlabel('Hrs/day')
plt.ylabel('Score')
plt.show()

#Finding Mean Absoulute Error
from sklearn import metrics
mean_abs_error = metrics.mean_absolute_error(y_test, prediction)

