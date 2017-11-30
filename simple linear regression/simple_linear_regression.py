# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 20:03:09 2017
Set Wd by Tools-> preferences -> Global working directory
@author: NirajS
"""

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =  pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#splitting the data into the training set and test set for the machine to learn from train and test it
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

"""
#feature scaling to transform the nominal data on to the same scale -1 to 1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#train must be fitted 1st and then test must be transformed
X_train = sc_X.fit_transform(X_train)  #fit calc the mean and the std deviatoin
X_test = sc_X.transform(X_test)        # we apply the same fit for the test as train
"""

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  #machine learns correlation on the training set for inde and de variables
regressor.fit(X_train,y_train)  

#predict the test set results
y_pred = regressor.predict(X_test)

#visualize the training set results 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualize the test set results 
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

