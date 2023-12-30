from pycoingecko import CoinGeckoAPI
from binance.client import Client
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import time
import concurrent.futures


class DataLoader:

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.start_datetime = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        self.end_datetime = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        self.symbols_list = self.get_tickers()
        self.df = self.combine_data()

    def get_tickers(self):
        cg = CoinGeckoAPI()
        exchanges = cg.get_exchanges_list()
        binance_id = next(exchange['id'] for exchange in exchanges if exchange['name'] == 'Binance')
        self.binance_tickers = cg.get_exchanges_tickers_by_id(binance_id)
        self.coins_id = [binance_ticker['coin_id'] for binance_ticker in self.binance_tickers['tickers']]
        symbols_list = [ticker['base'] + ticker['target'] for ticker in self.binance_tickers['tickers'] if
                        ticker['target'] == "USDT"]

        self.coin_id_to_ticker = {ticker['coin_id']: ticker['base'] + ticker['target'] for ticker in
                                  self.binance_tickers['tickers'] if ticker['target'] == "USDT"}
        self.ticker_to_coin_id = {value: key for key, value in self.coin_id_to_ticker.items()}
        return symbols_list

    '''def get_historical_market_caps(self):
        cg = CoinGeckoAPI()
        historical_data = {}
        start_date = self.convert_to_unix_timestamp(self.start_date)
        end_date = self.convert_to_unix_timestamp(self.end_date)

        for crypto_id in self.coins_id:
            market_caps = cg.get_coin_market_chart_range_by_id(id=crypto_id,
                                                               vs_currency='usd',
                                                               from_timestamp=start_date,
                                                               to_timestamp=end_date)['market_caps']
            historical_data[crypto_id] = [(datetime.fromtimestamp(ts / 1000).date(), cap) for ts, cap in market_caps]

        return historical_data

    # verifier historical data
    # trouver une correspondence entre les cols de market caps et les available tickers
    def market_caps(self):
        historical_data = self.get_historical_market_caps()
        dataframes = {}

        for crypto_id, data in historical_data.items():
            df = pd.DataFrame(data, columns=['date', crypto_id])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            dataframes[crypto_id] = df

        # Fusionner tous les DataFrames en un seul
        market_caps_df = pd.concat(dataframes.values(), axis=1)

        # Remplacer les valeurs NaN par des zéros
        market_caps_df.fillna(0, inplace=True)
        self.market_caps_df = market_caps_df

        renamed_columns = {crypto_id: self.coin_id_to_ticker.get(crypto_id, crypto_id) for crypto_id in
                           self.market_caps_df.columns}
        self.market_caps_df.rename(columns=renamed_columns, inplace=True)'''

    def get_market_caps(self, selected_symbols):
            market_caps_dict = {}
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {executor.submit(self.get_market_caps_for_symbol, symbol): symbol for symbol in selected_symbols}
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        market_caps = future.result()
                        df = pd.DataFrame(market_caps)
                        market_caps_dict[symbol] = df
                    except Exception as e:
                        print(f"Exception occurred while getting market caps for coin {symbol}: {e}")

            return market_caps_dict

    def get_market_caps_for_symbol(self, symbol):
        cg = CoinGeckoAPI()
        if symbol in self.symbols_list:
            coin_id = self.ticker_to_coin_id[symbol]
            market_data = cg.get_coin_market_chart_range_by_id(id=coin_id, vs_currency='usd',
                                                                   from_timestamp=self.convert_to_unix_timestamp(
                                                                       self.start_date),
                                                                   to_timestamp=self.convert_to_unix_timestamp(
                                                                       self.end_date))
            market_caps = []
            for timestamp, cap in market_data['market_caps']:
                date = datetime.fromtimestamp(timestamp / 1000)  # Convert the timestamp to datetime
                market_caps.append({'date': date, 'symbol': symbol, 'market_cap': cap})

            return market_caps

    def convert_to_unix_timestamp(self, date):
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        timestamp = int(date_obj.timestamp())
        return timestamp

    def get_data_for_coin(self, symbol):
        url = 'https://api.binance.com/api/v3/klines'
        req_params = {'symbol': symbol, 'interval': '1d', 'startTime': self.start_datetime,
                      'endTime': self.end_datetime}
        response = requests.get(url, params=req_params)
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
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.get_data_for_coin, symbol): symbol for symbol in self.symbols_list}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    df = future.result()
                    if df.shape != (0, 5):
                        data[symbol] = df
                except Exception as e:
                    print(f"Exception occurred while getting data for coin {symbol}: {e}")
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

    def __init__(self, data, rebalancing_window, initial_capital, start_date, end_date):
        self.data = data
        self.weights = [{coin: [0] for coin in data.symbols_list}]  # initialisation des poids à 0
        self.rebalancing_window = rebalancing_window
        self.initial_capital = initial_capital

        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        end_date -= timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d')

        dates = pd.date_range(start=start_date, end=end_date)
        self.portfolio = pd.DataFrame(0, index=dates, columns=data.symbols_list)
        self.portfolio_value = pd.DataFrame(0, index=dates, columns=['Value'])

        start_date_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        available_assets = [coin for coin in data.symbols_list if start_date_datetime in data.dataframes[coin].index]
        # ne repartir le acpital inital qu'entre les cryptos existentes au depart

        num_assets = len(available_assets)
        initial_allocation = initial_capital / num_assets

        for coin in data.symbols_list:
            allocation = initial_allocation if coin in available_assets else 0
            self.portfolio.at[start_date, coin] = allocation

        self.portfolio_value.at[start_date, 'Value'] = initial_capital

        self.last_rebalancing = None

        self.backtest = Backtest(self, data, start_date, end_date)
        self.backtest.run()

    def calculate_weights(self):

        self.available_assets = [coin for coin in self.data.symbols_list if
                                 self.backtest.current_date in self.data.dataframes[coin].index]
        current_market_caps = self.data.market_caps_df.loc[self.backtest.current_date]

        total_market_cap = current_market_caps[self.available_assets].sum()

        weights = {}
        for coin in self.data.symbols_list:
            if coin in self.available_assets:
                weights[coin] = current_market_caps[coin] / total_market_cap
            else:
                weights[coin] = 0
        return weights

    def calculate_returns(self, coin):
        current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
        self.previous_date = self.data.dataframes[coin].index[current_date_index - 1]
        previous_value = float(self.data.dataframes[coin].loc[self.previous_date, 'Close'])
        current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
        return np.log(current_value / previous_value)

    def maj_portfolio_value(self):

        for coin in self.available_assets:
            r = self.calculate_returns(coin)
            current_value = self.portfolio.loc[self.previous_date, coin] * (1 + r)
            self.portfolio.at[self.backtest.current_date, coin] = current_value

        total_portfolio_value = self.portfolio.loc[self.backtest.current_date].sum()
        self.portfolio_value.at[self.backtest.current_date, 'Value'] = total_portfolio_value

    def rebalancing(self):

        weights = self.calculate_weights()
        if self.backtest.current_date != datetime.strptime(start_date, '%Y-%m-%d'):

            self.weights.append(weights)
            self.maj_portfolio_value()

            for coin in self.available_assets:

                if self.portfolio.loc[self.previous_date, coin] != 0:
                    r = self.calculate_returns(coin)
                    current_value = self.portfolio.loc[self.previous_date, coin] * (
                            1 + r)  # problème ici, si première valeur = 0 ça se mettra pas à jour

                    self.portfolio.at[self.backtest.current_date, coin] = current_value
                    target = self.portfolio_value.loc[self.backtest.current_date, 'Value'] * self.weights[-1][coin]
                    # modifier pour avoir par rapport aux poids

                    if target > current_value:
                        self.go_long(coin)

                    elif target < current_value:
                        self.go_short(coin)
                else:
                    self.portfolio.at[self.backtest.current_date, coin] = self.portfolio_value.loc[
                                                                              self.backtest.current_date, 'Value'] * \
                                                                          self.weights[-1][coin]
                    # modifier ça

    def go_short(self, coin):

        r = self.calculate_returns(coin)
        new_value = self.portfolio.at[self.previous_date, coin] * (1 + r)
        target_value = self.weights[-1][coin] * self.portfolio_value.loc[self.backtest.current_date, 'Value']

        amount_to_sell = new_value - target_value
        self.portfolio.at[self.backtest.current_date, coin] -= amount_to_sell

    def go_long(self, coin):

        r = self.calculate_returns(coin)
        new_value = self.portfolio.at[self.previous_date, coin] * (1 + r)
        target_value = self.weights[-1][coin] * self.portfolio_value.loc[self.backtest.current_date, 'Value']

        amount_to_buy = target_value - new_value
        self.portfolio.at[self.backtest.current_date, coin] += amount_to_buy

    def apply_strategy(self):
        if self.last_rebalancing is None or (
                self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
            self.rebalancing()
            self.last_rebalancing = self.backtest.current_date
        else:
            for coin in self.available_assets:
                current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
                previous_date = self.data.dataframes[coin].index[current_date_index - 1]
                previous_value = float(self.data.dataframes[coin].loc[previous_date, 'Close'])
                current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])

                r = np.log(current_value) - np.log(previous_value)
                new_value = self.portfolio.loc[previous_date, coin] * (1 + r)
                self.portfolio.at[self.backtest.current_date, coin] = new_value
            self.portfolio_value.at[self.backtest.current_date, 'Value'] = self.portfolio.loc[
                self.backtest.current_date].sum()


class EqualWeightStrategy(AbstractStrategy):

    def __init__(self, data, rebalancing_window, initial_capital, start_date, end_date):
        self.data = data
        self.weights = [{coin: [0] for coin in data.symbols_list}]  # initialisation des poids à 0
        self.rebalancing_window = rebalancing_window
        self.initial_capital = initial_capital

        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        end_date -= timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d')

        dates = pd.date_range(start=start_date, end=end_date)
        self.portfolio = pd.DataFrame(0, index=dates, columns=data.symbols_list)
        self.portfolio_value = pd.DataFrame(0, index=dates, columns=['Value'])

        start_date_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        available_assets = [coin for coin in data.symbols_list if start_date_datetime in data.dataframes[coin].index]
        # ne repartir le acpital inital qu'entre les cryptos existentes au depart

        num_assets = len(available_assets)
        initial_allocation = initial_capital / num_assets

        for coin in data.symbols_list:
            allocation = initial_allocation if coin in available_assets else 0
            self.portfolio.at[start_date, coin] = allocation

        self.portfolio_value.at[start_date, 'Value'] = initial_capital

        self.last_rebalancing = None

        self.backtest = Backtest(self, data, start_date, end_date)
        self.backtest.run()

    def calculate_weights(self):

        self.available_assets = [coin for coin in self.data.symbols_list if
                                 self.backtest.current_date in self.data.dataframes[coin].index]
        num_assets = len(self.available_assets)
        weight = 1 / num_assets
        weights = {coin: weight for coin in self.available_assets}

        for coin in self.data.symbols_list:
            if coin not in self.available_assets:
                weights[coin] = 0

        return weights

    def calculate_returns(self, coin):
        current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
        self.previous_date = self.data.dataframes[coin].index[current_date_index - 1]
        previous_value = float(self.data.dataframes[coin].loc[self.previous_date, 'Close'])
        current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])
        r = np.log(current_value / previous_value)
        return r

    def maj_portfolio_value(self):

        for coin in self.available_assets:
            r = self.calculate_returns(coin)
            current_value = self.portfolio.loc[self.previous_date, coin] * (1 + r)
            self.portfolio.at[self.backtest.current_date, coin] = current_value

        total_portfolio_value = self.portfolio.loc[self.backtest.current_date].sum()
        self.portfolio_value.at[self.backtest.current_date, 'Value'] = total_portfolio_value

    def rebalancing(self):
        weights = self.calculate_weights()
        if self.backtest.current_date != datetime.strptime(start_date, '%Y-%m-%d'):

            self.weights.append(weights)
            self.maj_portfolio_value()

            for coin in self.available_assets:

                if self.portfolio.loc[self.previous_date, coin] != 0:
                    r = self.calculate_returns(coin)
                    current_value = self.portfolio.loc[self.previous_date, coin] * (
                            1 + r)  # problème ici, si première valeur = 0 ça se mettra pas à jour

                    self.portfolio.at[self.backtest.current_date, coin] = current_value
                    target = self.portfolio_value.loc[self.backtest.current_date, 'Value'] / len(self.available_assets)

                    if target > current_value:
                        self.go_long(coin)

                    elif target < current_value:
                        self.go_short(coin)
                else:
                    self.portfolio.at[self.backtest.current_date, coin] = self.portfolio_value.at[
                                                                              self.backtest.current_date, 'Value'] / len(
                        self.available_assets)

    def go_short(self, coin):

        r = self.calculate_returns(coin)
        new_value = self.portfolio.at[self.previous_date, coin] * (1 + r)
        target_value = self.weights[-1][coin] * self.portfolio_value.loc[self.backtest.current_date, 'Value']

        amount_to_sell = new_value - target_value
        self.portfolio.at[self.backtest.current_date, coin] -= amount_to_sell

    def go_long(self, coin):

        r = self.calculate_returns(coin)
        new_value = self.portfolio.at[self.previous_date, coin] * (1 + r)
        target_value = self.weights[-1][coin] * self.portfolio_value.loc[self.backtest.current_date, 'Value']

        amount_to_buy = target_value - new_value
        self.portfolio.at[self.backtest.current_date, coin] += amount_to_buy

    def apply_strategy(self):
        if self.last_rebalancing is None or (
                self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
            self.rebalancing()
            self.last_rebalancing = self.backtest.current_date
        else:
            for coin in self.available_assets:
                current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
                previous_date = self.data.dataframes[coin].index[current_date_index - 1]
                previous_value = float(self.data.dataframes[coin].loc[previous_date, 'Close'])
                current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])

                r = np.log(current_value) - np.log(previous_value)
                new_value = self.portfolio.loc[previous_date, coin] * (1 + r)
                self.portfolio.at[self.backtest.current_date, coin] = new_value

            self.portfolio_value.at[self.backtest.current_date, 'Value'] = self.portfolio.loc[
                self.backtest.current_date].sum()


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
        # Calcul des rendements quotidiens
        returns = portfolio_values['Value'].pct_change().dropna()
        return returns

    @staticmethod
    def cumulative_returns(portfolio_values):
        # Calcul des rendements cumulatifs
        cumulative_returns = (portfolio_values['Value'] / portfolio_values['Value'].iloc[0]) - 1
        return cumulative_returns

    @staticmethod
    def calculate_total_return(portfolio_values):
        # Calcul du rendement total
        total_return = (portfolio_values['Value'].iloc[-1] - portfolio_values['Value'].iloc[0]) / \
                       portfolio_values['Value'].iloc[0]
        return total_return  # pn d'annualisation

    @staticmethod
    def calculate_volatility(portfolio_values):
        # Calcul de la volatilité des rendements
        returns = PerformanceMetrics.calculate_returns(portfolio_values)
        volatility = returns.std()
        return volatility

    @staticmethod
    def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0):
        # Calcul des rendements et de la volatilité
        returns = PerformanceMetrics.calculate_total_return(portfolio_values)
        volatility = PerformanceMetrics.calculate_volatility(portfolio_values)

        # Calcul du ratio de Sharpe
        sharpe_ratio = ((returns - risk_free_rate) / (volatility * np.sqrt(364)))
        return sharpe_ratio

    @staticmethod
    def calculate_sortino_ratio(portfolio_values, risk_free_rate=0):
        expected_return = PerformanceMetrics.calculate_total_return(portfolio_values)
        returns = PerformanceMetrics.calculate_returns(portfolio_values)
        negative_returns = returns[returns < 0]  # Garde uniquement les rendements négatifs

        downside_deviation = negative_returns.std()

        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

        return sortino_ratio

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        values = portfolio_values['Value']
        peak = values.cummax()
        drawdown = (values - peak) / peak
        max_drawdown = drawdown.min()
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
        fig.suptitle('Dashboard de Performance', y=0.98, fontsize=20, fontweight='bold')
        plt.subplots_adjust(wspace=0.25)

        # Graphique des Valeurs du Portefeuille
        axs[0, 0].plot(portfolio_values['Value'], color='green', lw=2)
        axs[0, 0].set_title('Valeur du Portefeuille', fontweight='bold')
        axs[0, 0].set_ylabel('Valeur du Portefeuille')
        axs[0, 0].set_xlabel('Date')

        # Graphique des Rendements
        axs[0, 1].plot(returns, color='blue', lw=2)
        axs[0, 1].set_title('Rendements (%)', fontweight='bold')
        axs[0, 1].set_ylabel('Rendements')
        axs[0, 1].set_xlabel('Date')

        # Graphique des Rendements Cumulatifs
        axs[1, 0].plot(cumulative_returns, color='red', lw=2)
        axs[1, 0].set_title('Rendements Cumulatifs (%)', fontweight='bold')
        axs[1, 0].set_ylabel('Rendements Cumulatifs')
        axs[1, 0].set_xlabel('Date')

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
        table_data = [[col, metrics_df[col].values[0]] for col in metrics_df.columns]
        table = axs[1, 1].table(cellText=table_data, colLabels=["Métrique", "Valeur"], cellLoc="center", loc="center",
                                bbox=[0, 0, 1, 1])
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Titres des colonnes
                cell.set_text_props(weight='bold')

        plt.show()


if __name__ == '__main__':
    start_date = "2021-01-01"
    end_date = "2021-12-31"

    t1 = time.time()
    datas = DataLoader(start_date, end_date)  # 20 sec
    t2 = time.time()
    minutes, seconds = divmod(t2 - t1, 60)
    print(f'Time: {int(minutes)} minutes {seconds} seconds')

    t1 = time.time()
    market_caps = datas.get_market_caps(datas.symbols_list)  # 14 minutes
    t2 = time.time()
    minutes, seconds = divmod(t2 - t1, 60)
    print(f'Time: {int(minutes)} minutes {seconds} seconds')

    '''test = EqualWeightStrategy(datas, 30, 100000, start_date, end_date)
    r = PerformanceMetrics.calculate_returns(test.portfolio_value)
    PerformanceMetrics.stat_dashboard(test.portfolio_value, r, 1)'''

    '''test2 = MarketCapStrategy(datas, 30, 100000, start_date, end_date)
    r = PerformanceMetrics.calculate_returns(test2.portfolio_value)
    PerformanceMetrics.stat_dashboard(test2.portfolio_value, r, 1)'''

    '''###### import makret_caps
    start_date2 = "2019-01-01"
    end_date2 = "2023-12-01"
    
    datas2 = DataLoader(start_date2, end_date2)
    datas2.market_caps()'''
