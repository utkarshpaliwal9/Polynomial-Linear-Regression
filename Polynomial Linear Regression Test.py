# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 02:18:33 2017

@author: alien
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 10)
x_poly = poly.fit_transform(X)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_poly, Y)

#Visualise
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(x_poly), color = 'blue')
plt.xlabel('Level in Office')
plt.ylabel("Salary")
plt.title("Salary(Red) vs Predictions(Blue)")
plt.figure(figsize = (8,8) )
plt.show()