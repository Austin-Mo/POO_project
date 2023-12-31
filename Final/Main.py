# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:30:37 2023

@author: paul-
"""

from pycoingecko import CoinGeckoAPI
from binance.client import Client
import pandas as pd
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import calendar


start_date = "2021-12-30"
end_date = "2023-11-30"
market_caps = pd.read_csv("C:/Users/paul-/OneDrive/Documents/M2 Dauphine/M2/Python POO/market_caps_data.csv", index_col=0, parse_dates=True)
datas= DataLoader(start_date, end_date)
#datas.market_caps()
       
equiweighted = EqualWeightStrategy(datas, 10, 100000, start_date, end_date)
market_caps_strat = MarketCapStrategy(datas, market_caps, 10, 100000, start_date, end_date)
priceweighted = PriceWeightedStrategy(datas, 10, 10000, start_date, end_date)
PerformanceMetrics.stat_dashboard(market_caps_strat.portfolio_value)
PerformanceMetrics.stat_dashboard(equiweighted.portfolio_value)
PerformanceMetrics.stat_dashboard(priceweighted.portfolio_value)