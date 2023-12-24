from pycoingecko import CoinGeckoAPI
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime

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
        binance_tickers = cg.get_exchanges_tickers_by_id(binance_id)
        return [ticker['base'] + ticker['target'] for ticker in binance_tickers['tickers']]

    def get_data_for_coin(self, symbol):
        start_str = datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%d %b, %Y")
        end_str = datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%d %b, %Y")

        candles = self.binance_client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_str, end_str)
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore']
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


binance_api_key = 'aoVyyPRzDHHHZgxm0Fzb2s3FZ3aFot46ERv9bGSBHOb8O2G7BfvKtFEY51mXDeJ7'
binance_api_secret = 'WtAK2hkUwG2pNCayBlhlQmJPWdd9MqrQuZ62AOuky5g8o9LrEjQ66p8xQZnvLYZZ'

start_date = "2021-01-01"
end_date = "2021-12-31"

datas = DataLoader(binance_api_key, binance_api_secret, start_date, end_date)

# import des données avec l'api binance

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

    def calculate_weights(self, data):
        num_assets = len(data)
        weight = 1 / num_assets
        weights = {coin: weight for coin in data}
        return weights

    def apply_strategy(self, data):
        weights = self.calculate_weights(data)


class Backtest:

    def __init__(self, strategy, data_loader):
        self.strategy = strategy
        self.data_loader = data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = None

    def run(self):
        for current_date, daily_data in self.generate_data():
            self.current_date = current_date
            self.strategy.apply_strategy()

    def calculate_returns(self):
   # Calcul des rendements à partir de la valeur du portefeuille
       portfolio_values = np.array(self.strategy.portfolio_value)
       returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
       return returns

    def generate_data(self):
        for date, data in self.data_loader.df.loc[self.start_date:self.end_date].iterrows():
            yield date, data

        # Calcul du ratio de Sharpe à la fin du backtest
        returns = self.calculate_returns()
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(returns)
        print(f"Sharpe Ratio: {sharpe_ratio}")


class PerformanceMetrics:

    def calculate_total_return(portfolio_values):
       # Calcul du rendement total
       total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
       return total_return

    def calculate_volatility(returns):
       # Calcul de la volatilité
       volatility = np.std(returns)
       return volatility

    def calculate_sharpe_ratio(returns, risk_free_rate=0):
     # Calcul du rendement en excès
     excess_returns = returns - risk_free_rate

     # Calcul du rendement moyen et de l'écart type des rendements
     average_return = np.mean(excess_returns)
     volatility = np.std(excess_returns)

     # Calcul du ratio de Sharpe
     sharpe_ratio = (average_return / volatility) if volatility != 0 else 0

     return sharpe_ratio


class IndexCompositionTracker:

    def __init__(self):
        self.compositions = []

    def update_composition(self, date, composition):
        self.compositions.append((date, composition))
