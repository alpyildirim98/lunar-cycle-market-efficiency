#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:31:37 2023

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

full_moon = pd.read_csv("full_moon.csv")

full_moon.drop(full_moon.columns[[0, 2]], axis=1, inplace=True)

full_moon = full_moon[1478:1526]
full_moon.reset_index(drop=True, inplace=True)
full_moon['Dummy'] = 1
full_moon[0] = pd.to_datetime(full_moon[' Date'])
full_moon.set_index(0, inplace=True)
full_moon.drop(full_moon.columns[0], axis=1, inplace=True)

# Set the frequency of the index to daily
full_moon = full_moon.asfreq('D')

# Fill all missing values with 0
full_moon = full_moon.fillna(0)

# Forward fill the 1 values
full_moon = full_moon.ffill()

# Create a date range from 2019-07-06 to 2023-05-18
date_range = pd.date_range(start='2019-07-06', end='2023-05-18', freq='D')

# Reindex the dataframe with the new date range
full_moon = full_moon.reindex(date_range, fill_value=0)
full_moon.index = full_moon.index.date

# Create a new column for the shifted values
full_moon['shifted'] = full_moon['Dummy'].shift(-3)

# Create a new column for the rolling sum of the shifted values
full_moon['rolling_sum'] = full_moon['shifted'].rolling(window=7, min_periods=1).sum()

# Create a new column for the ones values
full_moon['ones'] = 0
full_moon.loc[full_moon['rolling_sum'] >= 1, 'ones'] = 1

# Drop the shifted and rolling_sum columns
full_moon = full_moon.drop(['shifted', 'rolling_sum'], axis=1)

# Create a DataFrame with zeros for all dates
january_effect = pd.DataFrame(0, index=date_range, columns=['Value'])

# Set the value to 1 for all dates in January
january_effect.loc[january_effect.index.month == 1, 'Value'] = 1

january_effect.index = january_effect.index.date

index_to_copy = full_moon.index
log_returns = log_returns.set_index(index_to_copy)
january_effect = january_effect.set_index(index_to_copy)

regression_data1 = pd.concat([log_returns, full_moon['ones']], axis=1, join='inner')
regression_data1 = pd.concat([regression_data1, january_effect['Value']], axis=1, join='inner')
regression_data1 = regression_data1.rename(columns={'Value': 'January'})
regression_data1 = regression_data1.rename(columns={'ones': 'Lunardummy'})

## Define dependent and independent variables | Portfolio
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['port_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

## Define dependent and independent variables | BTC
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['BTCUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_BTC = sm.OLS(y, X).fit()

# Print model summary
print(model_BTC.summary())

## Define dependent and independent variables | ETH
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['ETHUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_ETH = sm.OLS(y, X).fit()

# Print model summary
print(model_ETH.summary())

## Define dependent and independent variables | BNB
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['BNBUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_BNB = sm.OLS(y, X).fit()

# Print model summary
print(model_BNB.summary())


## Define dependent and independent variables | ADA
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['ADAUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_ADA = sm.OLS(y, X).fit()

# Print model summary
print(model_ADA.summary())



## Define dependent and independent variables | DOGE
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['DOGEUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_DOGE = sm.OLS(y, X).fit()

# Print model summary
print(model_DOGE.summary())

## Define dependent and independent variables | LINK
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['LINKUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_LINK = sm.OLS(y, X).fit()

# Print model summary
print(model_LINK.summary())

## Define dependent and independent variables | LTC
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['LTCUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_LTC = sm.OLS(y, X).fit()

# Print model summary
print(model_LTC.summary())


## Define dependent and independent variables | TRX
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['TRXUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_TRX = sm.OLS(y, X).fit()

# Print model summary
print(model_TRX.summary())


## Define dependent and independent variables | XMR
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['XMRUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_XMR = sm.OLS(y, X).fit()

# Print model summary
print(model_XMR.summary())


## Define dependent and independent variables | XRP
X = regression_data1[['Lunardummy', 'January']]
y = regression_data1[['XRPUSDT_return']]

# Add constant to independent variables
X = sm.add_constant(X)

# Fit linear regression model
model_XRP = sm.OLS(y, X).fit()

# Print model summary
print(model_XRP.summary())


# Create a list of the models
models = [model_ADA, model_BNB, model_BTC, model_DOGE, model_ETH, model_LINK, model_LTC, model_TRX, model_XMR, model_XRP, model]

# Use summary_col to create a table of regression results
results_table = summary_col(models, stars=True, float_format='%0.4f', model_names=['ADA', 'BNB', 'BTC', 'DOGE', 'ETH', 'LINK', 'LTC', 'TRX', 'XMR', 'XRP', 'Portfolio'])

# Print the table regression results
print(results_table)
