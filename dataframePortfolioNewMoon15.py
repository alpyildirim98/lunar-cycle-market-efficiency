#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:09:24 2023

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
symbols = ['ADAUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETHUSDT', 'LINKUSDT',
           'LTCUSDT', 'TRXUSDT', 'XMRUSDT', 'XRPUSDT']

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

close_prices.set_index(BTCUSDT['open_time'], inplace=True)
close_prices = close_prices.dropna()  

# Define an empty DataFrame to store the log returns
log_returns = pd.DataFrame()

# Loop through the symbols and read the corresponding CSV file
for symbol in symbols:
    filename = f'{symbol}.csv'
    df = pd.read_csv(filename)
    # Calculate the log returns and add them to the log_returns DataFrame with updated column names
    log_returns[symbol + '_return'] = np.log(df['close']).diff()

log_returns.set_index(BTCUSDT['open_time'], inplace=True)
log_returns = log_returns.dropna()  

# Define the weights for the portfolio
weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Calculate the portfolio log return
log_returns['port_return'] = log_returns.iloc[:, :10].values.dot(weights)

new_moon = pd.read_csv("new_moon.csv")

new_moon[0] = pd.to_datetime(new_moon['Date'])
new_moon.set_index(0, inplace=True)
new_moon.drop(new_moon.columns[0], axis=1, inplace=True)

# Set the frequency of the index to daily
new_moon = new_moon.asfreq('D')

# Fill all missing values with 0
new_moon = new_moon.fillna(0)

# Forward fill the 1 values
new_moon = new_moon.ffill()

# Create a date range from 2019-07-06 to 2023-05-18
date_range = pd.date_range(start='2019-07-06', end='2023-05-18', freq='D')

# Reindex the dataframe with the new date range
new_moon = new_moon.reindex(date_range, fill_value=0)
new_moon.index = new_moon.index.date

# Create a new column for the shifted values
new_moon['shifted'] = new_moon['Dummy'].shift(-7)

# Create a new column for the rolling sum of the shifted values
new_moon['rolling_sum'] = new_moon['shifted'].rolling(window=15, min_periods=1).sum()

# Create a new column for the ones values
new_moon['ones'] = 0
new_moon.loc[new_moon['rolling_sum'] >= 1, 'ones'] = 1

# Drop the shifted and rolling_sum columns
new_moon = new_moon.drop(['shifted', 'rolling_sum'], axis=1)
new_moon.loc[new_moon.index[:4], 'ones'] = 1
new_moon.loc[new_moon.index[-7:], 'ones'] = 1

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


## Define dependent and independent variables | BTC
X_BTC = regression_data1[['ones']]
y_BTC = regression_data1[['BTCUSDT_return']]

# Add constant to independent variables
X_BTC = sm.add_constant(X_BTC)

# Fit linear regression model
model_BTC = sm.OLS(y_BTC, X_BTC).fit()

# Print model summary
print(model_BTC.summary())


## Define dependent and independent variables | ETH
X_ETH = regression_data1[['ones']]
y_ETH = regression_data1[['ETHUSDT_return']]

# Add constant to independent variables
X_ETH = sm.add_constant(X_ETH)

# Fit linear regression model
model_ETH = sm.OLS(y_ETH, X_ETH).fit()

# Print model summary
print(model_ETH.summary())

## Define dependent and independent variables | BNB
X_BNB = regression_data1[['ones']]
y_BNB = regression_data1[['BNBUSDT_return']]

# Add constant to independent variables
X_BNB = sm.add_constant(X_BNB)

# Fit linear regression model
model_BNB = sm.OLS(y_BNB, X_BNB).fit()

# Print model summary
print(model_BNB.summary())


## Define dependent and independent variables | ADA
X_ADA = regression_data1[['ones']]
y_ADA = regression_data1[['ADAUSDT_return']]

# Add constant to independent variables
X_ADA = sm.add_constant(X_ADA)

# Fit linear regression model
model_ADA = sm.OLS(y_ADA, X_ADA).fit()

# Print model summary
print(model_ADA.summary())


## Define dependent and independent variables | DOGE
X_DOGE = regression_data1[['ones']]
y_DOGE = regression_data1[['DOGEUSDT_return']]

# Add constant to independent variables
X_DOGE = sm.add_constant(X_DOGE)

# Fit linear regression model
model_DOGE = sm.OLS(y_DOGE, X_DOGE).fit()

# Print model summary
print(model_DOGE.summary())

## Define dependent and independent variables | LINK
X_LINK = regression_data1[['ones']]
y_LINK = regression_data1[['LINKUSDT_return']]

# Add constant to independent variables
X_LINK = sm.add_constant(X_LINK)

# Fit linear regression model
model_LINK = sm.OLS(y_LINK, X_LINK).fit()

# Print model summary
print(model_LINK.summary())

## Define dependent and independent variables | LTC
X_LTC = regression_data1[['ones']]
y_LTC = regression_data1[['LTCUSDT_return']]

# Add constant to independent variables
X_LTC = sm.add_constant(X_LTC)

# Fit linear regression model
model_LTC = sm.OLS(y_LTC, X_LTC).fit()

# Print model summary
print(model_LTC.summary())

## Define dependent and independent variables | TRX
X_TRX = regression_data1[['ones']]
y_TRX = regression_data1[['TRXUSDT_return']]

# Add constant to independent variables
X_TRX = sm.add_constant(X_TRX)

# Fit linear regression model
model_TRX = sm.OLS(y_TRX, X_TRX).fit()

# Print model summary
print(model_TRX.summary())


## Define dependent and independent variables | XMR
X_XMR = regression_data1[['ones']]
y_XMR = regression_data1[['XMRUSDT_return']]

# Add constant to independent variables
X_XMR = sm.add_constant(X_XMR)

# Fit linear regression model
model_XMR = sm.OLS(y_XMR, X_XMR).fit()

# Print model summary
print(model_XMR.summary())


## Define dependent and independent variables | XRP
X_XRP = regression_data1[['ones']]
y_XRP = regression_data1[['XRPUSDT_return']]

# Add constant to independent variables
X_XRP = sm.add_constant(X_XRP)

# Fit linear regression model
model_XRP = sm.OLS(y_XRP, X_XRP).fit()

# Print model summary
print(model_XRP.summary())

# Create a list of the models
models = [model_ADA, model_BNB, model_BTC, model_DOGE, model_ETH, model_LINK, model_LTC, model_TRX, model_XMR, model_XRP, model]

# Use summary_col to create a table of regression results
results_table = summary_col(models, stars=True, float_format='%0.4f', model_names=['ADA', 'BNB', 'BTC', 'DOGE', 'ETH', 'LINK', 'LTC', 'TRX', 'XMR', 'XRP', 'Portfolio'])

# Print the table regression results
print(results_table)

