# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:30:37 2023

@author: paul-
"""
import pandas as pd
from Dataloader import DataLoader
from Strategy import EqualWeightStrategy, MarketCapStrategy, PriceWeightedStrategy
from Performance_Metrics import PerformanceMetrics

if __name__ == '__main__':
    start_date = "2019-01-30"
    end_date = "2023-11-30"
    market_caps = pd.read_csv("./market_caps_data.csv", index_col=0, parse_dates=True)
    datas = DataLoader(start_date, end_date)
    # datas.market_caps()

    equiweighted = EqualWeightStrategy(data=datas, rebalancing_window=10, initial_capital=100000,
                                       start_date=start_date, end_date=end_date)
    market_caps_strat = MarketCapStrategy(data=datas, market_caps=market_caps, rebalancing_window=10,
                                          initial_capital=100000, start_date=start_date, end_date=end_date)
    priceweighted = PriceWeightedStrategy(data=datas, rebalancing_window=10, initial_capital=100000,
                                          start_date=start_date, end_date=end_date)
    PerformanceMetrics.stat_dashboard(portfolio_values=market_caps_strat.portfolio_value, risk_free_rate=0)
    PerformanceMetrics.stat_dashboard(portfolio_values=market_caps_strat.portfolio_value, risk_free_rate=0)
    PerformanceMetrics.stat_dashboard(portfolio_values=market_caps_strat.portfolio_value, risk_free_rate=0)
