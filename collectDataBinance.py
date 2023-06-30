#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:40:39 2023

@author: alpyildirim
"""
import pandas as pd
from binance.client import Client
import datetime as dt
import pytz

# client configuration
api_key = 'bLxjTFZnzcrNvxnSnL0Wh61sopFvBf1nC3YEFWXkdFxVbTOKf5FesVzv1NZgkMm0' 
api_secret = 'phwRD92aiTshZBJGd2JeIonBhuiEdv8eFSfEuXcjZHFejFO5Yc5jtACP9L80YONq'
client = Client(api_key, api_secret)

# List of symbols
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "TRXUSDT", "LINKUSDT", "XMRUSDT"]

# Loop through the symbols and retrieve and save the data for each symbol
for symbol in symbols:
    interval= '1d'
    Client.KLINE_INTERVAL_1DAY
    klines = client.get_historical_klines(symbol, interval, "5 Jul,2019")
    df = pd.DataFrame(klines)
    df.columns = ["open_time","open", "high", "low", "close", "volume", "close_time", "qav","num_trades","taker_base_vol", "taker_quote_vol", "ignore"]

    # Convert the open_time and close_time columns to datetime objects
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    df = df.iloc[:, :6]
    df.set_index('open_time', inplace=True)

    df = df.astype(float)
    filename = f"{symbol}.csv"
    df.to_csv(filename)
    print(f"{symbol} data saved to {filename}")