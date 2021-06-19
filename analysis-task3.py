# -*- coding: utf-8 -*-

#Importing Packages
import numpy as ap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing Dataset
dataset = pd.read_csv('SampleSuperstore.csv')

'''
By Analyzing the dataset
Categorical Variables - Ship Mode, Segment, Country, City, State,
Postal Code, Region, Category, Sub-Category

Continuous Variables - Quantity, Discount

Output - Sales, Profit
'''


#Categorywise Analysis of Profit and Sales
dataset['Category'].value_counts()
plt.figure(figsize = (10, 8))
dataset['Category'].value_counts().plot.pie(autopct = '')
plt.show()

dataset.groupby('Category')[['Profit', 'Sales']].sum().plot.bar(color=['red','yellow'], alpha=0.9, figsize=(8,5))
plt.ylabel('Sales/Profit')
plt.show()
'''
Conclusion - Technology have highest sales, 
and Technology and Office Supplies have significantly larger profit than furniture
'''

#Sub-Categorywise analysis
dataset['Sub-Category'].value_counts()
plt.figure(figsize = (10, 8))
dataset['Sub-Category'].value_counts().plot.pie(autopct = '')
plt.show()

dataset.groupby('Sub-Category')[['Profit', 'Sales']].sum().plot.bar(color=['red','yellow'], alpha=0.9, figsize=(8,5))
plt.ylabel('Sales/Profit')
plt.show()
'''
Conclusion - Binders, Paper, Furnishings have high no. of buyers
Supplies, machines, copiers have lowest
Phones and Chairs have largest number of sales, copiers have highest profit
Tables and Bookcases are going in loss
'''

#State wise Analysis
dataset['State'].value_counts()
plt.figure(figsize = (10, 8))
sns.countplot(x='State', data = dataset, order=dataset['State'].value_counts().index)
plt.xticks(rotation=90)
plt.show()

dataset.groupby('State')[['Profit', 'Sales']].sum().plot.bar(color=['red','yellow'], alpha=0.9, figsize=(8,5))
plt.ylabel('Sales/Profit')
plt.show()
'''
Conclusion - Highest no. of buyers in California
highes sales and profit in California and New york
Loss in Texas, Ohio, Illinios
'''

#Region wise Analysis
dataset['Region'].value_counts()
plt.figure(figsize = (10, 8))
dataset['Region'].value_counts().plot.pie(autopct = '')
plt.show()

dataset.groupby('Region')[['Profit', 'Sales']].sum().plot.bar(color=['red','yellow'], alpha=0.9, figsize=(8,5))
plt.ylabel('Sales/Profit')
plt.show()

plt.figure(figsize = (12,8))
plt.title('Segment wise Sales in each Region')
sns.barplot(x='Region',y='Sales',data=dataset,hue='Segment',order=dataset['Region'].value_counts().index)
plt.xlabel('Region')
plt.show()
'''
Conclusion - Segmentwise sales are almost same in each region
East and West Region have max profits
South have min profits
'''

#Segment wise Analysis
dataset['Segment'].value_counts()
plt.figure(figsize = (10, 8))
dataset['Segment'].value_counts().plot.pie(autopct = '')
plt.show()

dataset.groupby('Segment')[['Profit', 'Sales']].sum().plot.bar(color=['red','yellow'], alpha=0.9, figsize=(8,5))
plt.ylabel('Sales/Profit')
plt.show()
'''
Conclusion - Profit and Sales max in Consumer Segment, so the buyers
Min in Home Office
'''

#Ship Mode Analysis
dataset['Ship Mode'].value_counts()
plt.figure(figsize = (10, 8))
sns.countplot(x='Ship Mode', data = dataset, order=dataset['Ship Mode'].value_counts().index)
plt.xticks(rotation=90)
plt.show()


#Profit and Sales analysis with Discount
sns.lineplot(x='Discount',y='Profit',data=dataset,label='Profit')
plt.legend()
plt.show()

sns.lineplot(x='Discount',y='Sales',data=dataset,label='Sales')
plt.legend()
plt.show()

'''
Conclusion - No correlation of Profit and Sales with Discount
'''

#Profit analysis with Quantity
year
'''
Conclusion - no specific correlation
'''

#Analyzing the number of Buyers
dataset.hist(figsize=(10,10), bins=50)
plt.show()

'''
Conclusion - Most no. of buyers buy Quantity of 2 and 3
Most no. of discount is given between 0 to 20 percent
'''










