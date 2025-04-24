#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Basic Exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns='Cabin', inplace=True) 

# Encode Categorical Variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Outlier Detection & Removal
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()

# Remove outliers
fare_threshold = 300
df = df[df['Fare'] < fare_threshold]

# Saving cleaned data
df.to_csv("titanic_cleaned.csv", index=False)
print("Data cleaned and saved to 'titanic_cleaned.csv'")


# In[ ]:




