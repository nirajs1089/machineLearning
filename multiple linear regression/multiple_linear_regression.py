# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:47:47 2017

@author: NirajS
"""

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =  pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


#encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#encoded the states to 0,1,2 which are ordinal 
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
#to avoid ordinality, [dummy variables] we create binary state for each country into different columns from rows
oneHotEncoder = OneHotEncoder(categorical_features=[3]) #idx of the column
X = oneHotEncoder.fit_transform(X).toarray() 

#avoiding the dummy variable trap by dropping 1 column 
X = X[:,1:]


#splitting the data into the training set and test set for the machine to learn from train and test it
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

"""
#feature scaling to transform the nominal data on to the same scale -1 to 1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#train must be fitted 1st and then test must be transformed
X_train = sc_X.fit_transform(X_train)  #fit calc the mean and the std deviatoin
X_test = sc_X.transform(X_test)        # we apply the same fit for the test as train
"""

#fiting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predictin of the test results for profit
y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()