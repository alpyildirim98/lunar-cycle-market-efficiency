#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 01:52:28 2023

@author: alpyildirim
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dtime
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# Define a list of symbols
symbols = ["KAVAUSDT", "YFIUSDT", "ANKRUSDT", "BALUSDT", "COMPUSDT", "RUNEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "UNIUSDT"]

# Loop through the symbols and read the corresponding CSV file
for symbol in symbols:
    filename = f'{symbol}.csv'
    df = pd.read_csv(filename)
    # Name the dataframe according to its symbol
    globals()[symbol] = df
    
# Define an empty DataFrame to store the close prices
close_prices = pd.DataFrame()

# Loop through the symbols and read the corresponding CSV file
for symbol in symbols:
    filename = f'{symbol}.csv'
    df = pd.read_csv(filename)
    # Add the close prices to the close_prices DataFrame
    close_prices[symbol + '_close'] = df['close']

close_prices.set_index(ANKRUSDT['open_time'], inplace=True)
close_prices = close_prices.dropna()  

# Define an empty DataFrame to store the log returns
log_returns = pd.DataFrame()

# Loop through the symbols and read the corresponding CSV file
for symbol in symbols:
    filename = f'{symbol}.csv'
    df = pd.read_csv(filename)
    # Calculate the log returns and add them to the log_returns DataFrame with updated column names
    log_returns[symbol + '_return'] = np.log(df['close']).diff()

log_returns.set_index(ANKRUSDT['open_time'], inplace=True)
log_returns = log_returns.dropna()  

# Define the weights for the portfolio
weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Calculate the portfolio log return
log_returns['port_return'] = log_returns.iloc[:, :10].values.dot(weights)

new_moon = pd.read_csv("new_moon.csv")

new_moon = new_moon[16:]
new_moon.reset_index(drop=True, inplace=True)
new_moon[0] = pd.to_datetime(new_moon['Date'])
new_moon.set_index(0, inplace=True)
new_moon.drop(new_moon.columns[0], axis=1, inplace=True)

# Set the frequency of the index to daily
new_moon = new_moon.asfreq('D')

# Fill all missing values with 0
new_moon = new_moon.fillna(0)

# Forward fill the 1 values
new_moon = new_moon.ffill()

# Create a date range from 2020-09-18 to 2023-05-20
date_range = pd.date_range(start='2020-09-18', end='2023-05-20', freq='D')

# Reindex the dataframe with the new date range
new_moon = new_moon.reindex(date_range, fill_value=0)
new_moon.index = new_moon.index.date

# Create a new column for the shifted values
new_moon['shifted'] = new_moon['Dummy'].shift(-3)

# Create a new column for the rolling sum of the shifted values
new_moon['rolling_sum'] = new_moon['shifted'].rolling(window=7, min_periods=1).sum()

# Create a new column for the ones values
new_moon['ones'] = 0
new_moon.loc[new_moon['rolling_sum'] >= 1, 'ones'] = 1

# Drop the shifted and rolling_sum columns
new_moon = new_moon.drop(['shifted', 'rolling_sum'], axis=1)
new_moon.loc[new_moon.index[:3], 'ones'] = 1

index_to_copy = new_moon.index
log_returns = log_returns.set_index(index_to_copy)

regression_data1 = pd.concat([log_returns, new_moon['ones']], axis=1, join='inner')


## Define dependent and independent variables | Portfolio
X = regression_data1[['ones']]
y = regression_data1[['port_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())


## Define dependent and independent variables | KAVA
X_KAVA = regression_data1[['ones']]
y_KAVA = regression_data1[['KAVAUSDT_return']]

# Add constant to independent variables
X_KAVA = sm.add_constant(X_KAVA)

# Fit linear regression model
model_KAVA = sm.OLS(y_KAVA, X_KAVA).fit()

# Print model summary
print(model_KAVA.summary())


## Define dependent and independent variables | YFI
X_YFI = regression_data1[['ones']]
y_YFI = regression_data1[['YFIUSDT_return']]

# Add constant to independent variables
X_YFI = sm.add_constant(X_YFI)

# Fit linear regression model
model_YFI = sm.OLS(y_YFI, X_YFI).fit()

# Print model summary
print(model_YFI.summary())

## Define dependent and independent variables | ANKR
X_ANKR = regression_data1[['ones']]
y_ANKR = regression_data1[['ANKRUSDT_return']]

# Add constant to independent variables
X_ANKR = sm.add_constant(X_ANKR)

# Fit linear regression model
model_ANKR = sm.OLS(y_ANKR, X_ANKR).fit()

# Print model summary
print(model_ANKR.summary())


## Define dependent and independent variables | BAL
X_BAL = regression_data1[['ones']]
y_BAL = regression_data1[['BALUSDT_return']]

# Add constant to independent variables
X_BAL = sm.add_constant(X_BAL)

# Fit linear regression model
model_BAL = sm.OLS(y_BAL, X_BAL).fit()

# Print model summary
print(model_BAL.summary())


## Define dependent and independent variables | COMP
X_COMP = regression_data1[['ones']]
y_COMP = regression_data1[['COMPUSDT_return']]

# Add constant to independent variables
X_COMP = sm.add_constant(X_COMP)

# Fit linear regression model
model_COMP = sm.OLS(y_COMP, X_COMP).fit()

# Print model summary
print(model_COMP.summary())

## Define dependent and independent variables | RUNE
X_RUNE = regression_data1[['ones']]
y_RUNE = regression_data1[['RUNEUSDT_return']]

# Add constant to independent variables
X_RUNE = sm.add_constant(X_RUNE)

# Fit linear regression model
model_RUNE = sm.OLS(y_RUNE, X_RUNE).fit()

# Print model summary
print(model_RUNE.summary())

## Define dependent and independent variables | MKR
X_MKR = regression_data1[['ones']]
y_MKR = regression_data1[['MKRUSDT_return']]

# Add constant to independent variables
X_MKR = sm.add_constant(X_MKR)

# Fit linear regression model
model_MKR = sm.OLS(y_MKR, X_MKR).fit()

# Print model summary
print(model_MKR.summary())

## Define dependent and independent variables | SNX
X_SNX = regression_data1[['ones']]
y_SNX = regression_data1[['SNXUSDT_return']]

# Add constant to independent variables
X_SNX = sm.add_constant(X_SNX)

# Fit linear regression model
model_SNX = sm.OLS(y_SNX, X_SNX).fit()

# Print model summary
print(model_SNX.summary())


## Define dependent and independent variables | CRV
X_CRV = regression_data1[['ones']]
y_CRV = regression_data1[['CRVUSDT_return']]

# Add constant to independent variables
X_CRV = sm.add_constant(X_CRV)

# Fit linear regression model
model_CRV = sm.OLS(y_CRV, X_CRV).fit()

# Print model summary
print(model_CRV.summary())


## Define dependent and independent variables | UNI
X_UNI = regression_data1[['ones']]
y_UNI = regression_data1[['UNIUSDT_return']]

# Add constant to independent variables
X_UNI = sm.add_constant(X_UNI)

# Fit linear regression model
model_UNI = sm.OLS(y_UNI, X_UNI).fit()

# Print model summary
print(model_UNI.summary())

# Create a list of the models
models = [model_KAVA, model_YFI, model_ANKR, model_BAL, model_COMP, model_RUNE, model_MKR, model_SNX, model_CRV, model_UNI, model]

# Use summary_col to create a table of regression results
results_table = summary_col(models, stars=True, float_format='%0.4f', model_names=['KAVA', 'YFI', 'ANKR', 'BAL', 'COMP', 'RUNE', 'MKR', 'SNX', 'CRV', 'UNI', 'Portfolio'])

# Print the table regression results
print(results_table)