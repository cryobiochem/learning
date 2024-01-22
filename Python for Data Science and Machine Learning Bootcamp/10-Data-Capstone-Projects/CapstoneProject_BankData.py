import numpy as np
import pandas as pd
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns


# Optional Plotly Method Imports
import plotly as py
import cufflinks as cf
cf.go_offline()






bank_stocks = pd.read_pickle("C:\\Users\\Asus\\github\\testspace\\Learning\\Python for Data Science and Machine Learning Bootcamp\\10-Data-Capstone-Projects\\all_banks")
bank_stocks

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

# What is the max Close price for each bank's stock throughout the time period?
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()


# Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:
returns = pd.DataFrame()
for i in tickers:
    returns[i] = bank_stocks[i]['Close'].pct_change()


# Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?
sns.pairplot(returns)


# on what dates each bank stock had the best and worst single day returns
returns.idxmin()


# Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? Which would you classify as the riskiest for the year 2015?
returns.std()
returns.ix['2015-01-01':'2015-12-31'].std()


# Create a distplot using seaborn of the 2015 returns for Morgan Stanley
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS'], bins=100, color='green')


# Create a distplot using seaborn of the 2008 returns for CitiGroup
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C'], bins=100, color='red')



# Create a line plot showing Close price for each bank for the entire index of time.
close = pd.DataFrame()
for i in tickers:
    close[i] = bank_stocks[i]['Close']

close.iplot()


# Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008
plt.figure(figsize=(12,4))
bank_stocks['BAC']['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 day Mov AVG')
bank_stocks['BAC']['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC Close 2008')
plt.legend()


# Create a heatmap of the correlation between the stocks Close Price.
sns.heatmap(close.corr())

# Use seaborn's clustermap to cluster the correlations together:
sns.clustermap(close.corr())


# analyze trends of temporal data
bac15 = bank_stocks['BAC'][['Open', 'High', 'Low', 'Close']].loc['2015-01-01':'2016-01-01']
bac15.iplot('candle')


# study a simple 7, 14, 30-day moving average with cufflinks
# moving averages allow to predict whats going to happen based on past data
bank_stocks['MS']['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma', periods=[7,14,30])


# Bollinger Plot shows stdev of stock price as it moves up through time
bank_stocks['BAC']['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll')
