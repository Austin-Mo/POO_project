# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:59:39 2023

@author: paul-
"""
from pycoingecko import CoinGeckoAPI
from binance.client import Client
import pandas as pd
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class DataLoader:

    def __init__(self, binance_api_key, binance_api_secret, start_date, end_date):
        self.binance_client = Client(binance_api_key, binance_api_secret)
        self.start_date = start_date
        self.end_date = end_date
        self.symbols_list = self.get_tickers()
        self.df = self.combine_data()

    def get_tickers(self):
        cg = CoinGeckoAPI()
        exchanges = cg.get_exchanges_list()
        binance_id = next(exchange['id'] for exchange in exchanges if exchange['name'] == 'Binance')
        self.binance_tickers = cg.get_exchanges_tickers_by_id(binance_id)
        self.coins_id = [binance_ticker['coin_id'] for binance_ticker in self.binance_tickers['tickers']]
        symbols_list =  [ticker['base'] + ticker['target'] for ticker in self.binance_tickers['tickers'] if ticker['target'] == "USDT"]
        return symbols_list
    
    def get_historical_market_caps(self):
        cg = CoinGeckoAPI()
        historical_data = {}
        start_date = self.convert_to_unix_timestamp(self.start_date)
        end_date = self.convert_to_unix_timestamp(self.end_date)
    
        for crypto_id in self.coins_id:
            market_caps = cg.get_coin_market_chart_range_by_id(id=crypto_id, 
                                                               vs_currency='usd', 
                                                               from_timestamp=start_date, 
                                                               to_timestamp=end_date)['market_caps']
            
            historical_data[crypto_id] = [(datetime.fromtimestamp(ts/1000).date(), cap) for ts, cap in market_caps]
    
        return historical_data
    
    def market_caps(self):
        self.market_caps_df = self.get_historical_market_caps()

    def convert_to_unix_timestamp(self, date):
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        timestamp = int(date_obj.timestamp())
        return timestamp

    def get_data_for_coin(self, symbol):
        start_str = datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%d %b, %Y")
        end_str = datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%d %b, %Y")
        
        candles = self.binance_client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_str, end_str)
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore']
        df = pd.DataFrame(candles, columns=columns)

        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def get_data(self):
        data = {}
        for symbol in self.symbols_list:
            if self.get_data_for_coin(symbol).shape != (0,5):
                data[symbol] = self.get_data_for_coin(symbol)
        self.symbols_list = list(data.keys())
        return data

    def combine_data(self):
        self.dataframes = self.get_data()
        combined_df = pd.concat(self.dataframes.values(), axis=1, keys=self.dataframes.keys())
        combined_df.columns = ['_'.join(col).strip() for col in combined_df.columns.values]
        return combined_df
    


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
        self.portfolio = {coin: [initial_capital/len(data.symbols_list)] for coin in data.symbols_list}  # pareil ici je veux un df avec des dates
        self.portfolio_value = [initial_capital] #remplacer ici car je ne veux pas une liste mais un df avec des dates pour correspondance 
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

                close_value = self.data.dataframes[coin].loc[self.backtest.current_date, 'Close']
                #print(f'close value is {close_value} and its type is {type(close_value)}')
                if self.weights[-1][coin] > float(close_value)/self.portfolio_value[-1]:
                    self.go_long(coin)
                elif self.weights[-1][coin] < float(close_value)/self.portfolio_value[-1]:
                    self.go_short(coin)
                
        total_portfolio_value = sum(self.portfolio[coin][-1] for coin in self.portfolio) # pas bon ici car parfois 'est reutiliser plusieurs fois
        print(f"total_portfolio_value is {total_portfolio_value}")
        self.portfolio_value.append(total_portfolio_value)
     
        
    def go_short(self, coin):
        current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
        previous_date = self.data.dataframes[coin].index[current_date_index - 1]
        previous_value = float(self.data.dataframes[coin].loc[previous_date, 'Close'])
        current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
        
        r = np.log(current_value) - np.log(previous_value)
        
        current_value = self.portfolio[coin][-1] * (1+r)
        target_value = self.weights[-1][coin] * self.portfolio_value[-1]
        amount_to_sell = current_value - target_value
        
        self.portfolio[coin].append(self.portfolio[coin][-1] - amount_to_sell)
        
    def go_long(self, coin):
        current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
        previous_date = self.data.dataframes[coin].index[current_date_index - 1]
        previous_value = float(self.data.dataframes[coin].loc[previous_date, 'Close'])
        current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
        
        r = np.log(current_value) - np.log(previous_value)
        
        current_value = self.portfolio[coin][-1] * (1+r)
        target_value = self.weights[-1][coin] * self.portfolio_value[-1]
        amount_to_buy = target_value - current_value
        
        self.portfolio[coin].append(self.portfolio[coin][-1] + amount_to_buy)
        
    def apply_strategy(self):
        if self.last_rebalancing is None or (self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
            self.rebalancing()
            self.last_rebalancing = self.backtest.current_date

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
    def calculate_returns(portfolio_values):
        # Calcul du rendement total
        portfolio_values = np.array(portfolio_values)
        returns = np.log(portfolio_values[1:] / portfolio_values[:-1])
        return returns
    
    @staticmethod
    def cumulative_returns(portfolio_values):
        portfolio_values = pd.Series(portfolio_values)
        cumulative_returns = ((portfolio_values / portfolio_values.iloc[0]) - 1).cumsum()
        return cumulative_returns
    
    @staticmethod
    def calculate_total_return(portfolio_values):
        # Calcul du rendement total
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        return total_return
    
    @staticmethod
    def calculate_volatility(portfolio_values):
        # Calcul de la volatilité
        returns = PerformanceMetrics.calculate_returns(portfolio_values)
        volatility = np.std(returns)
        return volatility
    
    @staticmethod
    def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0):
        returns = PerformanceMetrics.calculate_total_return(portfolio_values)

        # Calcul du rendement espéré et de l'écart type des rendements
        volatility = PerformanceMetrics.calculate_volatility(portfolio_values)

        # Calcul du ratio de Sharpe
        sharpe_ratio = ((returns - risk_free_rate) / volatility) if volatility != 0 else 0

        return sharpe_ratio
    
    @staticmethod
    def calculate_sortino_ratio(portfolio_values, risk_free_rate=0):
        expected_return = PerformanceMetrics.calculate_total_return(portfolio_values)
        returns = PerformanceMetrics.calculate_returns(portfolio_values)
        negative_returns = np.minimum(returns, 0)  # Garde les rendements négatifs, rendements positifs remplacés par 0

        downside_deviation = np.std(negative_returns)

        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

        return sortino_ratio
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values):
   
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)
        return max_drawdown
    
    @staticmethod
    def calculate_annualized_return(portfolio_values, num_years):
        total_return = PerformanceMetrics.calculate_total_return(portfolio_values)
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        return annualized_return
    
    @staticmethod
    def stat_dashboard(portfolio_values, returns, num_years, risk_free_rate=0):
        # Calcul des métriques
        cumulative_returns = PerformanceMetrics.cumulative_returns(portfolio_values)
        total_return = PerformanceMetrics.calculate_total_return(portfolio_values)
        annualized_return = PerformanceMetrics.calculate_annualized_return(portfolio_values, num_years)
        volatility = PerformanceMetrics.calculate_volatility(portfolio_values)
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(portfolio_values, risk_free_rate)
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(portfolio_values)
        sortino_ratio = PerformanceMetrics.calculate_sortino_ratio(portfolio_values, risk_free_rate)
        
        # Création du Dashboard
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.4})
        fig.suptitle('Dashboard de Performance', y=0.98, fontsize = 20, fontweight='bold')
        
        # Ajustement de l'espacement entre le titre de l'axe des ordonnées du graphe en haut à droite et le graphe en haut à gauche
        plt.subplots_adjust(wspace=0.25)
    
        # Graphique des Valeurs du Portefeuille
        axs[0, 0].plot(portfolio_values, color = 'green', lw = 2)
        axs[0, 0].set_title('Valeur du Portefeuille', fontweight='bold')
        axs[0, 0].set_ylabel('Portfolio value')
        axs[0, 0].set_xlabel('Days')
    
        # Graphique des Rendements
        axs[0, 1].plot(returns)
        axs[0, 1].set_title('Rendements (%)', fontweight='bold')
        axs[0, 1].set_ylabel('Returns')
        axs[0, 1].set_xlabel('Days')
        
        #Graphique des rendements cumulés
        axs[1, 0].plot(cumulative_returns, color = 'red', lw = 2)
        axs[1, 0].set_title('Rendements cumulés (%)', fontweight='bold')
        axs[1, 0].set_ylabel('Cumulative returns')
        axs[1, 0].set_xlabel('Days')

        # Tableau des Métriques de Performance
        axs[1, 1].axis('off')
        
        metrics_data = {
            "Rendement Total (%)": [round(total_return * 100, 2)],
            "Rendement Annualisé (%)": [round(annualized_return * 100, 2)],
            "Volatilité (%)": [round(volatility * 100, 2)],
            "Ratio de Sharpe": [round(sharpe_ratio, 2)],
            "Ratio de Sortino": [round(sortino_ratio, 2)],
            "Drawdown Max": [round(max_drawdown, 2)]
        }

        metrics_df = pd.DataFrame(metrics_data)
        
        table_data = []
        for col in metrics_df.columns:
            table_data.append([col, metrics_df[col].values[0]])
        
        # Centrer le tableau
        table = axs[1, 1].table(cellText=table_data, colLabels=["Métrique", "Valeur"], cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        
        # Mettre en gras les titres des colonnes
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # La première ligne correspond aux titres des colonnes
                cell.set_text_props(weight='bold')
        
        #Montrer le dashboard
        plt.show()
    
    
################################################################################# 
 
binance_api_key = 'aoVyyPRzDHHHZgxm0Fzb2s3FZ3aFot46ERv9bGSBHOb8O2G7BfvKtFEY51mXDeJ7'
binance_api_secret = 'WtAK2hkUwG2pNCayBlhlQmJPWdd9MqrQuZ62AOuky5g8o9LrEjQ66p8xQZnvLYZZ'

start_date = "2023-01-01"
end_date = "2023-12-20"

datas = DataLoader(binance_api_key, binance_api_secret, start_date, end_date)

test = EqualWeightStrategy(datas, 1, 100000, start_date, end_date)
r = PerformanceMetrics.calculate_returns(test.portfolio_value)
PerformanceMetrics.stat_dashboard(test.portfolio_value,r,1)

plt.plot(datas.dataframes["BTCUSDT"]["Close"])
plt.plot(r)

####################################################################################"

class IndexCompositionTracker:

    def __init__(self):
        self.compositions = []

    def update_composition(self, date, composition):
        self.compositions.append((date, composition))
        
        
class PriceWeightedStrategy(AbstractStrategy):
    
    def __init__(self, data, rebalancing_window, initial_capital, start_date, end_date):
        self.data = data
        self.weights = [{coin: [0] for coin in data.symbols_list}] # initialisation des poids à 0
        self.rebalancing_window = rebalancing_window
        self.initial_capital = initial_capital
        self.portfolio = {coin: [initial_capital/len(data.symbols_list)] for coin in data.symbols_list}  # pareil ici je veux un df avec des dates
        self.portfolio_value = [initial_capital] #remplacer ici car je ne veux pas une liste mais un df avec des dates pour correspondance 
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
            #print(self.weights[-1][coin])
            #print(self.weights[-2][coin])
            
            if self.backtest.current_date in self.data.dataframes[coin].index:

                close_value = self.data.dataframes[coin].loc[self.backtest.current_date, 'Close']
                #print(f'close value is {close_value} and its type is {type(close_value)}')
                if self.weights[-1][coin] > float(close_value)/self.portfolio_value[-1]:
                    self.go_long(coin)
                elif self.weights[-1][coin] < float(close_value)/self.portfolio_value[-1]:
                    self.go_short(coin)
                
        total_portfolio_value = sum(self.portfolio[coin][-1] for coin in self.portfolio) # pas bon ici car parfois 'est reutiliser plusieurs fois
        print(f"total_portfolio_value is {total_portfolio_value}")
        self.portfolio_value.append(total_portfolio_value)
     
        
    def go_short(self, coin):
        current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
        previous_date = self.data.dataframes[coin].index[current_date_index - 1]
        previous_value = float(self.data.dataframes[coin].loc[previous_date, 'Close'])
        current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
    
        r = np.log(current_value) - np.log(previous_value)
    
        current_value = self.portfolio[coin][-1] * (1 + r)
        target_value = self.weights[-1][coin] * self.portfolio_value[-1]
        amount_to_sell = current_value - target_value
    
        self.portfolio[coin].append(self.portfolio[coin][-1] - amount_to_sell)
    
    def go_long(self, coin):
        current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
        previous_date = self.data.dataframes[coin].index[current_date_index - 1]
        previous_value = float(self.data.dataframes[coin].loc[previous_date, 'Close'])
        current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
    
        r = np.log(current_value) - np.log(previous_value)
    
        current_value = self.portfolio[coin][-1] * (1 + r)
        target_value = self.weights[-1][coin] * self.portfolio_value[-1]
        amount_to_buy = target_value - current_value
    
        self.portfolio[coin].append(self.portfolio[coin][-1] + amount_to_buy)

        
    def apply_strategy(self):
        if self.last_rebalancing is None or (self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
            self.rebalancing()
            self.last_rebalancing = self.backtest.current_date
             
test2 = PriceWeightedStrategy(datas, 1, 100000, start_date, end_date)
r2 = PerformanceMetrics.calculate_returns(test2.portfolio_value)
PerformanceMetrics.stat_dashboard(test2.portfolio_value,r2,1) 
    
