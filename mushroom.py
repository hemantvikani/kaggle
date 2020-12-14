# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:27:54 2020

@author: heman
"""

import numpy as np # linear algebra
import pandas as pd # data pre-processing


df1=pd.read_csv("mushroom.csv")

df1

df1.columns

df1.info()

df1.isnull().sum()           #check null values in data

df1["class"].value_counts()   #count of both classes(poisonous and edible)

import seaborn as sns

sns.countplot(x="class",data=df1)

feature_names=df1.columns

feature_names

# Creating independent and dependent variables
x = df1.iloc[:,1:].values
y = df1.iloc[:,0].values         #labels

print(x)

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
new_y=encoder.fit_transform(y)


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(x).toarray()

x


from sklearn.model_selection import train_test_split  #splitting the data for training and validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
from sklearn.decomposition import PCA      #feature selection
pca = PCA(n_components = 3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

x_test


from sklearn.ensemble import RandomForestClassifier    #classification algorithm



rand_clf = RandomForestClassifier(random_state=6)


rc = rand_clf.fit(x_train,y_train)

print(rc.score(x_test,y_test))     #accuracy