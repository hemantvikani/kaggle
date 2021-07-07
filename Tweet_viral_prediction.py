# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:40:50 2021

@author: power
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

df = pd.read_csv("twitter.csv")

df= df.drop(['Unnamed: 0','Id','Post Contet','Published DateTime'], axis = 1)

print(df.columns)

print(df.head(5))

print(df.shape)

# Missing Values
print(df.isna().sum())

#Correlation between features

corr = df.corr()



plt.pyplot.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# Separate feature variables and target variable
X = df.drop(['Impact'], axis = 1)
y = df['Impact']

print(X['Media Type'].nunique())
 
#One hot encoded the Media Type column into 3 categories
X=pd.get_dummies(X)

# Normalize feature variables 
from sklearn.preprocessing import StandardScaler
X_features = X
X = StandardScaler().fit_transform(X)

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0,shuffle=True)


y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)


  
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

reg = LinearRegression(normalize=True)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
y_pred1= reg.predict(X_train)
r2score= r2_score(y_test,y_pred)
r2score1=r2_score(y_train,y_pred1)

#train and test score = R2 score and r2 score1
print("training score:",r2score1)
print("testing score:",r2score)

#Resultss
#training score: 0.999999995418001
#testing score: 0.9999999952531569


#mae
from sklearn import model_selection
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(reg, X, y, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))

#training error is the averge difference between the predicted and actual values
from sklearn.metrics import mean_squared_error
training_error = mean_squared_error(y_train,y_pred1)
testing_error = mean_squared_error(y_test,y_pred)

print("training error: ", training_error)

print("testing error1: ", testing_error)

#error results
#training error:  48.010003777250134
#testing error1:  52.51257948606291

print("As we can see there is not a much difference between training and testing score and also between training and testing error value, so this is a best fit model, there is no underfitting and overfitting")


from xgboost import XGBRegressor
# Define the xgb regression model
xgb = XGBRegressor(n_estimators=100,learning_rate=0.5,random_state=0)

# Fit the model

xgb.fit(X_train,y_train)


# Get predictions
y_pred_xgb = xgb.predict(X_test)
y_pred1_xgb= xgb.predict(X_train)
r2score_xgb_testing = r2_score(y_test,y_pred_xgb)
r2score_xgb_training=r2_score(y_train,y_pred1_xgb)


print("training score_xgb:",r2score_xgb_testing)
print("testing score_xgb:",r2score_xgb_training)

#Results
#training score_xgb: 0.9991009152430016
#testing score_xgb: 0.9999668728925873



from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(random_state=1,max_depth=20,n_estimators=20)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
y_pred1_RF = RF.predict(X_train)

RF_score1= r2_score(y_test,y_pred_RF)

RF_score = r2_score(y_train,y_pred1_RF)

print("training score_RF:",RF_score1)
print("testing score_RF:",RF_score)

#results
#training score_RF: 0.9994320947780388
#testing score_RF: 0.9998642646495634


#best score is of linear regression(#training score: 0.999999995418001
#testing score: 0.9999999952531569)



