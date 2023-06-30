#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:05:20 2023

@author: alpyildirim
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a list of symbols
symbols = ['ADAUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'TRXUSDT', 'XMRUSDT', 'XRPUSDT']

# Loop through the symbols and read the corresponding CSV file
for symbol in symbols:
    filename = f'{symbol}.csv'
    df = pd.read_csv(filename)
    # Name the dataframe according to its symbol
    globals()[symbol] = df

# Create a 5x2 grid of subplots
fig, axs = plt.subplots(5, 2, figsize=(10, 20))

# Plot each chart in a different subplot
axs[0, 0].plot(ADAUSDT["close"])
axs[0, 1].plot(BNBUSDT["close"])
axs[1, 0].plot(BTCUSDT["close"])
axs[1, 1].plot(DOGEUSDT["close"])
axs[2, 0].plot(ETHUSDT["close"])
axs[2, 1].plot(LINKUSDT["close"])
axs[3, 0].plot(LTCUSDT["close"])
axs[3, 1].plot(TRXUSDT["close"])
axs[4, 0].plot(XMRUSDT["close"])
axs[4, 1].plot(XRPUSDT["close"])



# Add titles and labels to each subplot
axs[0, 0].set_title('ADA')
axs[0, 1].set_title('BNB')
axs[1, 0].set_title('BTC')
axs[1, 1].set_title('DOGE')
axs[2, 0].set_title('ETH')
axs[2, 1].set_title('LINK')
axs[3, 0].set_title('LTC')
axs[3, 1].set_title('TRX')
axs[4, 0].set_title('XMR')
axs[4, 1].set_title('XRP')

# Display the plot
plt.show()