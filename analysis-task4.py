# -*- coding: utf-8 -*-

#importing Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Dataset
dataset = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='latin1')

dataset.head()

#Yearly Analysis of Terror
dataset['iyear'].value_counts()
plt.figure(figsize = (10, 8))
sns.countplot(x='iyear', data = dataset)
plt.xticks(rotation=90)
plt.show()

#Monthly Analysis of Terror
dataset['imonth'].value_counts()
plt.figure(figsize = (10, 8))
sns.countplot(x='imonth', data = dataset)
plt.xticks(rotation=90)
plt.show()

#Countrywise Analysis
dataset['country_txt'].value_counts()
plt.figure(figsize = (10, 8))
sns.countplot(x='country_txt', data = dataset, order=dataset['country_txt'].value_counts().index)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize = (10, 8))
dataset['country_txt'].value_counts().plot.pie(autopct = '')
plt.show()

#Citywise Analysis in Iraq, the most terrored city
dataset_iraq = dataset[dataset['country_txt'] == 'Iraq']
dataset_iraq['city'].value_counts()
dataset['city'].value_counts()
plt.figure(figsize = (10, 8))
dataset_iraq['city'].value_counts().plot.pie(autopct = '')
plt.show()

#Analyzing Success Rate
dataset_iraq['success'].value_counts()
dataset['success'].value_counts()

#Analyzing Attack Type
dataset['attacktype1_txt'].value_counts()
plt.figure(figsize = (10, 8))
sns.countplot(x='attacktype1_txt', data = dataset)
plt.xticks(rotation=90)
plt.show()

#Analyzing the Country where the most number of Hijacking is happening
dataset_hijack = dataset[dataset["attacktype1_txt"] == "Hijacking"]
dataset_hijack['country_txt'].value_counts()
sns.countplot(x='country_txt', data = dataset_hijack, order=dataset_hijack['country_txt'].value_counts().index)
plt.xticks(rotation=90)
plt.show()

#Analyzing Targets
dataset['targtype1_txt'].value_counts()
plt.figure(figsize = (10, 8))
sns.countplot(x='targtype1_txt', data = dataset)
plt.xticks(rotation=90)
plt.show()

#Analyzing Gang names
dataset['gname'].value_counts()

#Taliban Gang attacked which country the most?
dataset_taliban = dataset[dataset['gname']=="Taliban"]
dataset_taliban['country_txt'].value_counts()
plt.figure(figsize = (10, 8))
dataset_taliban['country_txt'].value_counts().plot.pie(autopct = '')
plt.show()

#Weapon Analytics and Weapon used by Taliban gang the most
dataset['weaptype1_txt'].value_counts()
plt.figure(figsize = (10, 8))
dataset['weaptype1_txt'].value_counts().plot.pie(autopct = '')
plt.show()

dataset_taliban['weaptype1_txt'].value_counts()
plt.figure(figsize = (10, 8))
dataset_taliban['weaptype1_txt'].value_counts().plot.pie(autopct = '')
plt.show()

#Property Value Analysis
dataset['propextent_txt'].value_counts()
plt.figure(figsize = (10, 8))
dataset['propextent_txt'].value_counts().plot.pie(autopct = '')
plt.show()

#Catastrophic Property loss analysis
dataset_catastrophic = dataset[dataset['propextent_txt']=="Catastrophic (likely >= $1 billion)"]
dataset_catastrophic["country_txt"].value_counts()

#Catastrophic Property Yearwise Analysis
dataset['iyear'].value_counts()
dataset.groupby('iyear')[['propvalue']].sum().plot.bar(color=['red'], alpha=0.9, figsize=(8,5))
plt.ylabel('Property value')
plt.show()

sns.lineplot(x='iyear',y='propvalue',data=dataset,label='Prop Value')
plt.legend()
plt.show()

#DB Source Analysis
dataset['dbsource'].value_counts()
plt.figure(figsize = (10, 8))
dataset['dbsource'].value_counts().plot.pie(autopct = '')
plt.show()


