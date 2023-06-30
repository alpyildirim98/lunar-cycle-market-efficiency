#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:14:40 2023

@author: alpyildirim
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a list of symbols
symbols = ["KAVAUSDT", "YFIUSDT", "ANKRUSDT", "BALUSDT", "COMPUSDT", "RUNEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "UNIUSDT"]

# Loop through the symbols and read the corresponding CSV file
for symbol in symbols:
    filename = f'{symbol}.csv'
    df = pd.read_csv(filename)
    # Name the dataframe according to its symbol
    globals()[symbol] = df

# Create a 5x2 grid of subplots
fig, axs = plt.subplots(5, 2, figsize=(10, 20))

# Plot each chart in a different subplot
axs[0, 0].plot(KAVAUSDT["close"])
axs[0, 1].plot(YFIUSDT["close"])
axs[1, 0].plot(ANKRUSDT["close"])
axs[1, 1].plot(BALUSDT["close"])
axs[2, 0].plot(COMPUSDT["close"])
axs[2, 1].plot(RUNEUSDT["close"])
axs[3, 0].plot(MKRUSDT["close"])
axs[3, 1].plot(SNXUSDT["close"])
axs[4, 0].plot(CRVUSDT["close"])
axs[4, 1].plot(UNIUSDT["close"])



# Add titles and labels to each subplot
axs[0, 0].set_title('KAVA')
axs[0, 1].set_title('YFI')
axs[1, 0].set_title('ANKR')
axs[1, 1].set_title('BAL')
axs[2, 0].set_title('COMP')
axs[2, 1].set_title('RUNE')
axs[3, 0].set_title('MKR')
axs[3, 1].set_title('SNX')
axs[4, 0].set_title('CRV')
axs[4, 1].set_title('UNI')

# Display the plot
plt.show()