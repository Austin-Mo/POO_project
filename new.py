# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 18:49:14 2023

@author: FABIEN
"""

from pycoingecko import CoinGeckoAPI
import requests
from binance.client import Client
import pandas as pd
from datetime import datetime
import numpy as np

class DataLoader:

    def __init__(self, start_date, end_date):
        self.start_date = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        self.end_date = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        self.cg = CoinGeckoAPI()
        self.symbols_list, self.id_list = self.get_tickers()
        self.df = self.combine_data()

    def get_tickers(self):
        exchanges = self.cg.get_exchanges_list()
        binance_id = next(exchange['id'] for exchange in exchanges if exchange['name'] == 'Binance')
        binance_tickers = self.cg.get_exchanges_tickers_by_id(binance_id)
        symbols_list = [ticker['base'] + ticker['target'] for ticker in binance_tickers['tickers']]
        id_list = [ticker['coin_id'] for ticker in binance_tickers['tickers']]
        return symbols_list, id_list

    def get_market_caps(self, selected_symbols):
        market_caps = []
        for symbol in selected_symbols:
            if symbol in self.symbols_list:
                coin_id = self.id_list[self.symbols_list.index(symbol)]
                market_data = self.cg.get_coin_market_chart_range_by_id(id=coin_id, vs_currency='usd',
                                                                        from_timestamp=self.start_date,
                                                                        to_timestamp=self.end_date)
                for timestamp, cap in market_data['market_caps']:
                    date = datetime.fromtimestamp(timestamp/1000)  # Convert the timestamp to datetime
                    market_caps.append({'date': date, 'symbol': symbol, 'market_cap': cap})

        df = pd.DataFrame(market_caps)
        return df

    def get_data_for_coin(self, symbol):
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&startTime={self.start_date}' \
              f'&endTime={self.end_date}'
        response = requests.get(url)
        candles = response.json()

        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                   'Number of trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore']
        df = pd.DataFrame(candles, columns=columns)

        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df

    def get_data(self):
        data = {}
        for symbol in self.symbols_list:
            data[symbol] = self.get_data_for_coin(symbol)
        return data

    def combine_data(self):
        self.dataframes = self.get_data()
        combined_df = pd.concat(self.dataframes.values(), axis=1, keys=self.dataframes.keys())
        combined_df.columns = ['_'.join(col).strip() for col in combined_df.columns.values]
        return combined_df


start_date = "2021-01-01"
end_date = "2021-12-31"

datas = DataLoader(start_date, end_date)

#cg = CoinGeckoAPI()
#market_caps = cg.get_coin_market_chart_range_by_id(id="bitcoin",vs_currency = "usd", from_timestamp="1609459200", to_timestamp="1640908800")['market_caps']

datas = DataLoader(start_date, end_date)
#datas.coins_id
#datas.get_historical_market_caps()
### import des données avec l'api binance

from abc import ABC, abstractmethod

class AbstractStrategy(ABC):
    
    @abstractmethod
    def calculate_weights(self):
        pass

    @abstractmethod
    def apply_strategy(self):
        pass
    

### Attention pour cette strategie il faut la marketcap
class MarketCapStrategy(AbstractStrategy):
    
    def __init__(self, data, market_caps, rebalancing_window, initial_capital, start_date, end_date):
        self.data = data
        self.market_caps = market_caps
        self.weights = [{coin: [0] for coin in data.get_tickers()}] # initialisation des poids à 0
        self.rebalancing_window = rebalancing_window
        self.initial_capital = initial_capital
        self.portfolio = {coin: [0] for coin in data.get_tickers()} 
        self.portfolio_value = [initial_capital]
        self.last_rebalancing = None
        self.backtest = Backtest(self, data, start_date, end_date)
        self.backtest.run()

    def calculate_weights(self):
        total_market_cap = sum(self.market_caps.values())
        weights = {coin: cap / total_market_cap for coin, cap in self.market_caps.items()}
        return weights

    def rebalancing(self):
        weights = self.calculate_weights()
        self.weights.append(weights) 
        
        for coin in self.data.get_tickers():
            if self.weights[-1][coin] > self.weights[-2][coin]:
                self.go_long(coin)
            elif self.weights[-1][coin] < self.weights[-2][coin]:
                self.go_short(coin)
        self.portfolio_value.append(sum(self.portfolio[coin][-1] for coin in self.portfolio))
    
    def go_short(self, coin):
        amount_to_sell = self.portfolio_value[-1] * self.weights[-1][coin] - self.portfolio_value[-1] * self.weights[-2][coin]
        self.portfolio[coin].append(amount_to_sell)
        
    def go_long(self, coin):
        amount_to_buy = self.portfolio_value[-1] * self.weights[-1][coin] - self.portfolio_value[-1] * self.weights[-2][coin]
        self.portfolio[coin].append(amount_to_buy)
        
    def apply_strategy(self):
        if self.last_rebalancing is None or (self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
            self.rebalancing()
            self.last_rebalancing = self.backtest.current_date
  
        
  
class EqualWeightStrategy(AbstractStrategy):
    
    def __init__(self, data, rebalancing_window, initial_capital, start_date, end_date):
        self.data = data
        self.weights = [{coin: [0] for coin in data.symbols_list}] # initialisation des poids à 0
        self.rebalancing_window = rebalancing_window
        self.initial_capital = initial_capital
        self.portfolio = {coin: [0] for coin in data.symbols_list} 
        self.portfolio_value = [initial_capital]
        self.last_rebalancing = None
        self.backtest = Backtest(self, data, start_date, end_date)
        self.backtest.run()

    def calculate_weights(self):
        num_assets = len(self.data.symbols_list)
        weight = 1 / num_assets
        weights = {coin: weight for coin in self.data.symbols_list}
        return weights

    def rebalancing(self):
        weights = self.calculate_weights()
        self.weights.append(weights) 
        
        for coin in self.data.symbols_list:
            #print(self.weights[-1][coin])
            #print(self.weights[-2][coin])
            
            if self.backtest.current_date in self.data.dataframes[coin].index:
                print(self.backtest.current_date)
                
                close_value = self.data.dataframes[coin].loc[self.backtest.current_date, 'Close']
                #print(f'close value is {close_value} and its type is {type(close_value)}')
                if self.weights[-1][coin] > float(close_value)/self.portfolio_value[-1]:
                    self.go_long(coin)
                elif self.weights[-1][coin] < float(close_value)/self.portfolio_value[-1]:
                    self.go_short(coin)
                self.portfolio_value.append(sum(self.portfolio[coin][-1] for coin in self.portfolio))
            else:
                print(f"Date {self.backtest.current_date} not found for {coin}")
     
    def go_short(self, coin):
        current_value = self.portfolio[coin][-1] * float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
        target_value = self.weights[-1][coin] * self.portfolio_value[-1]
        amount_to_sell = current_value - target_value
        
        self.portfolio[coin].append(self.portfolio[coin][-1] - amount_to_sell)
        
    def go_long(self, coin):
        current_value = self.portfolio[coin][-1] * float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
        target_value = self.weights[-1][coin] * self.portfolio_value[-1]
        amount_to_buy = target_value - current_value
        
        self.portfolio[coin].append(self.portfolio[coin][-1] + amount_to_buy)
        
    def apply_strategy(self):
        if self.last_rebalancing is None or (self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
            self.rebalancing()
            self.last_rebalancing = self.backtest.current_date
            
test = EqualWeightStrategy(datas, 10, 100000, start_date, end_date)
        
        
        
class Backtest:

    def __init__(self, strategy, data_loader, start_date, end_date):
        self.strategy = strategy
        self.data_loader = data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = None

    def run(self):
        for current_date, daily_data in self.generate_data():
            self.current_date = current_date
            self.strategy.apply_strategy()

    def generate_data(self):
        for date, data in self.data_loader.df.loc[self.start_date:self.end_date].iterrows():
            yield date, data
        
        

class PerformanceMetrics:
    @staticmethod
    def calculate_total_return(portfolio_values):
        # Calcul du rendement total
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        return total_return
    @staticmethod
    def calculate_returns(portfolio_values):
        # Calcul des rendements quotidiens
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return returns
    @staticmethod
    def calculate_volatility(returns):
        # Calcul de la volatilité
        volatility = np.std(returns)
        return volatility
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0):
        # Calcul du rendement en excès
        excess_returns = returns - risk_free_rate

        # Calcul du rendement moyen et de l'écart type des rendements
        average_return = np.mean(excess_returns)
        volatility = np.std(excess_returns)

        # Calcul du ratio de Sharpe
        sharpe_ratio = (average_return / volatility) if volatility != 0 else 0

        return sharpe_ratio

PerformanceMetrics.calculate_total_return(test.portfolio_value)

   
class IndexCompositionTracker:

    def __init__(self):
        self.compositions = []

    def update_composition(self, date, composition):
        self.compositions.append((date, composition))
        
        
        

        
        
        
class PriceWeightedStrategy(AbstractStrategy):
     
     def __init__(self, data, rebalancing_window, initial_capital, start_date, end_date):
         self.data = data
         self.weights = [{coin: [0] for coin in data.symbols_list}]  # initialization of weights to 0
         self.rebalancing_window = rebalancing_window
         self.initial_capital = initial_capital
         self.portfolio = {coin: [0] for coin in data.symbols_list} 
         self.portfolio_value = [initial_capital]
         self.last_rebalancing = None
         self.backtest = Backtest(self, data, start_date, end_date)
         self.backtest.run()

     def calculate_weights(self):
        # Calculate weights based on the closing prices of assets
        weights = {coin: float(self.data.dataframes[coin].iloc[-1]['Close']) for coin in self.data.symbols_list}
        
        # Normalize weights to ensure the sum is equal to 1
        total_price = sum(weights.values())
        weights = {coin: price / total_price for coin, price in weights.items()}
        
        return weights

     def rebalancing(self):
         weights = self.calculate_weights()
         self.weights.append(weights) 
         
         for coin in self.data.symbols_list:
             if self.backtest.current_date in self.data.dataframes[coin].index:
                 close_value = self.data.dataframes[coin].loc[self.backtest.current_date, 'Close']
                 
                 if self.weights[-1][coin] > float(close_value) / self.portfolio_value[-1]:
                     self.go_long(coin)
                 elif self.weights[-1][coin] < float(close_value) / self.portfolio_value[-1]:
                     self.go_short(coin)
                 
                 self.portfolio_value.append(sum(self.portfolio[coin][-1] for coin in self.portfolio))
             else:
                 print(f"Date {self.backtest.current_date} not found for {coin}")
      
     def go_short(self, coin):
         current_value = self.portfolio[coin][-1] * float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
         target_value = self.weights[-1][coin] * self.portfolio_value[-1]
         amount_to_sell = current_value - target_value
         
         self.portfolio[coin].append(self.portfolio[coin][-1] - amount_to_sell)
         
     def go_long(self, coin):
         current_value = self.portfolio[coin][-1] * float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
         target_value = self.weights[-1][coin] * self.portfolio_value[-1]
         amount_to_buy = target_value - current_value
         
         self.portfolio[coin].append(self.portfolio[coin][-1] + amount_to_buy)
         
     def apply_strategy(self):
         if self.last_rebalancing is None or (self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
             self.rebalancing()
             self.last_rebalancing = self.backtest.current_date
    