from pycoingecko import CoinGeckoAPI
from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import tkinter.messagebox as messagebox
import time


class DataLoader:

    def __init__(self, start_date, end_date):
         self.binance_client = Client('G4ocQV5RGywtoowV8YgvBjVWhA1shFKPrWzjqMfCgd7pBNEJxsTtZzbr1dNR6Sn2',
                                      '4xWeaVfyiWTmS0kAsdp03xSqZfu0Z42LYqj9LmSESiFYYTYyoWmXBQwSzjdN2Tol')
         self.start_date = start_date
         self.end_date = end_date
         #self.start_datetime = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()*1000)
         #self.end_datetime = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()*1000)
         self.symbols_list = self.get_tickers()
         self.df = self.combine_data()


    def get_tickers(self):
        cg = CoinGeckoAPI()
        exchanges = cg.get_exchanges_list()
        binance_id = next(exchange['id'] for exchange in exchanges if exchange['name'] == 'Binance')
        self.binance_tickers = cg.get_exchanges_tickers_by_id(binance_id)
        self.coins_id = [binance_ticker['coin_id'] for binance_ticker in self.binance_tickers['tickers']]
        symbols_list = [ticker['base'] + ticker['target'] for ticker in self.binance_tickers['tickers']
                        if ticker['target'] == "USDT"]

        self.coin_id_to_ticker = {ticker['coin_id']: ticker['base'] + ticker['target'] for
                                  ticker in self.binance_tickers['tickers'] if ticker['target'] == "USDT"}

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
    #verifier historical data
    #trouver une correspondence entre les cols de market caps et les available tickers
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

        renamed_columns = {crypto_id: self.coin_id_to_ticker.get(crypto_id, crypto_id)
                           for crypto_id in self.market_caps_df.columns}
        self.market_caps_df.rename(columns=renamed_columns, inplace=True)


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
        self.weights = [{coin: [0] for coin in data.symbols_list}] # initialisation des poids à 0
        self.rebalancing_window = rebalancing_window
        self.initial_capital = initial_capital

        self.start_date = start_date

        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        end_date -= timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d')

        dates = pd.date_range(start=start_date, end=end_date)
        self.portfolio = pd.DataFrame(0, index=dates, columns=data.symbols_list)
        self.portfolio_value = pd.DataFrame(0, index=dates, columns=['Value'])

        start_date_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        available_assets = [coin for coin in data.symbols_list if start_date_datetime in data.dataframes[coin].index]
        #ne repartir le acpital inital qu'entre les cryptos existentes au depart

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

        self.available_assets = [coin for coin in self.data.symbols_list if self.backtest.current_date in self.data.dataframes[coin].index]
        current_market_caps = market_caps.loc[self.backtest.current_date]

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
        r = np.log(current_value / previous_value)
        return r

    def maj_portfolio_value(self):

        for coin in self.available_assets:
            r = self.calculate_returns(coin)
            current_value = self.portfolio.loc[self.previous_date, coin] * (1+r)
            self.portfolio.at[self.backtest.current_date, coin] = current_value

        total_portfolio_value = self.portfolio.loc[self.backtest.current_date].sum()
        self.portfolio_value.at[self.backtest.current_date, 'Value'] = total_portfolio_value


    def rebalancing(self):
        weights = self.calculate_weights()
        if self.backtest.current_date != datetime.strptime(self.start_date, '%Y-%m-%d'):

            self.weights.append(weights)
            self.maj_portfolio_value()

            for coin in self.available_assets:

                if self.portfolio.loc[self.previous_date, coin] != 0:
                    r = self.calculate_returns(coin)
                    current_value = self.portfolio.loc[self.previous_date, coin] * (1+r) #problème ici, si première valeur = 0 ça se mettra pas à jour

                    self.portfolio.at[self.backtest.current_date, coin] = current_value
                    target = self.portfolio_value.loc[self.backtest.current_date, 'Value']*self.weights[-1][coin]
                    #modifier pour avoir par rapport aux poids

                    if target > current_value:
                        self.go_long(coin)

                    elif target < current_value:
                        self.go_short(coin)
                else:
                    self.portfolio.at[self.backtest.current_date, coin] = self.portfolio_value.loc[self.backtest.current_date, 'Value']*self.weights[-1][coin]
                    #modifier ça

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
        if self.last_rebalancing is None or (self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
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


            self.portfolio_value.at[self.backtest.current_date, 'Value'] = self.portfolio.loc[self.backtest.current_date].sum()

class EqualWeightStrategy(AbstractStrategy):

    def __init__(self, data, rebalancing_window, initial_capital, start_date, end_date, selected_cryptos):
        self.data = data
        self.weights = [{coin: [0] for coin in data.symbols_list}]  # initialisation des poids à 0
        self.rebalancing_window = rebalancing_window
        self.initial_capital = initial_capital

        self.selected_cryptos = selected_cryptos
        self.start_date = start_date

        #end_date = datetime.strptime(end_date, '%Y-%m-%d')
        #end_date -= timedelta(days=1)
        #end_date = end_date.strftime('%Y-%m-%d')

        dates = pd.date_range(start=start_date, end=end_date)
        self.portfolio = pd.DataFrame(0, index=dates, columns=data.symbols_list)
        self.portfolio_value = pd.DataFrame(0, index=dates, columns=['Value'])

        start_date_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        available_assets = [coin for coin in data.symbols_list if start_date_datetime in data.dataframes[coin].index]
        #ne repartir le acpital inital qu'entre les cryptos existentes au depart

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

        self.available_assets = [coin for coin in self.selected_cryptos if
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
            current_value = self.portfolio.loc[self.previous_date, coin] * (1+r)
            self.portfolio.at[self.backtest.current_date, coin] = current_value

        total_portfolio_value = self.portfolio.loc[self.backtest.current_date].sum()
        self.portfolio_value.at[self.backtest.current_date, 'Value'] = total_portfolio_value


    def rebalancing(self):
        weights = self.calculate_weights()
        if self.backtest.current_date != datetime.strptime(self.start_date, '%Y-%m-%d'):

            self.weights.append(weights)
            self.maj_portfolio_value()

            for coin in self.available_assets:

                if self.portfolio.loc[self.previous_date, coin] != 0:
                    r = self.calculate_returns(coin)
                    current_value = self.portfolio.loc[self.previous_date, coin] * (1+r) #problème ici, si première valeur = 0 ça se mettra pas à jour

                    self.portfolio.at[self.backtest.current_date, coin] = current_value
                    target = self.portfolio_value.loc[self.backtest.current_date, 'Value']/len(self.available_assets)

                    if target > current_value:
                        self.go_long(coin)

                    elif target < current_value:
                        self.go_short(coin)
                else:
                    self.portfolio.at[self.backtest.current_date, coin] = self.portfolio_value.at[self.backtest.current_date, 'Value']/len(self.available_assets)



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
        if self.last_rebalancing is None or (self.backtest.current_date - self.last_rebalancing).days >= self.rebalancing_window:
            self.rebalancing()
            self.last_rebalancing = self.backtest.current_date
        else:
            self.available_assets = [coin for coin in self.data.symbols_list if self.backtest.current_date in self.data.dataframes[coin].index]
            for coin in self.available_assets:
                current_date_index = self.data.dataframes[coin].index.get_loc(self.backtest.current_date)
                previous_date = self.data.dataframes[coin].index[current_date_index - 1]
                previous_value = float(self.data.dataframes[coin].loc[previous_date, 'Close'])
                current_value = float(self.data.dataframes[coin].loc[self.backtest.current_date, 'Close'])

                r = np.log(current_value) - np.log(previous_value)
                new_value = self.portfolio.loc[previous_date, coin] * (1 + r)
                self.portfolio.at[self.backtest.current_date, coin] = new_value

            self.portfolio_value.at[self.backtest.current_date, 'Value'] = self.portfolio.loc[self.backtest.current_date].sum()



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
    def calculate_log_returns(portfolio_values):
        log_returns = np.log(portfolio_values['Value'] / portfolio_values['Value'].shift(1)).dropna()
        return log_returns

    @staticmethod
    def cumulative_returns(portfolio_values):
        cumulative_returns = (portfolio_values['Value'] / portfolio_values['Value'].iloc[0]) - 1
        return cumulative_returns

    @staticmethod
    def calculate_total_return(portfolio_values):
        # Calcul du rendement total
        total_return = (portfolio_values['Value'].iloc[-1] - portfolio_values['Value'].iloc[0]) / portfolio_values['Value'].iloc[0]
        return total_return

    @staticmethod
    def calculate_volatility(portfolio_values):
        # Calcul de la volatilité des rendements
        returns = PerformanceMetrics.calculate_log_returns(portfolio_values)
        volatility = returns.std()
        return volatility*np.sqrt(365)

    @staticmethod
    def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0):
        
        returns = PerformanceMetrics.calculate_log_returns(portfolio_values)
        returns = (returns.mean())*365
        volatility = PerformanceMetrics.calculate_volatility(portfolio_values)
        sharpe_ratio = ((returns - risk_free_rate) / volatility)
        return sharpe_ratio

    @staticmethod
    def calculate_sortino_ratio(portfolio_values, risk_free_rate=0):
        expected_return = PerformanceMetrics.calculate_log_returns(portfolio_values)
        expected_return = (expected_return.mean())*365
        returns = PerformanceMetrics.calculate_log_returns(portfolio_values)
        negative_returns = returns[returns < 0]  # Garde uniquement les rendements négatifs

        downside_deviation = (negative_returns.std()) * np.sqrt(365)

        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

        return sortino_ratio

    @staticmethod
    def calculate_drawdown(portfolio_values):
        # Calcul des drawdowns
        peak = portfolio_values['Value'].cummax()
        drawdown = (portfolio_values['Value'] - peak) / peak
        return drawdown

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        values = portfolio_values['Value']
        peak = values.cummax()
        drawdown = (values - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    @staticmethod
    def calculate_annualized_return(portfolio_values):
        returns = PerformanceMetrics.calculate_log_returns(portfolio_values)
        annualized_return = (returns.mean())*365
        return annualized_return

    @staticmethod
    def plot_cumulative_returns(cumulative_returns, ax):
        ax.plot(cumulative_returns, color='green', lw=2)
        ax.set_title('Rendements Cumulatifs (%)', fontweight='bold')
        ax.set_ylabel('Rendements Cumulatifs')
        ax.set_xlabel('Date')

    @staticmethod
    def plot_returns(returns, ax):
        ax.plot(returns, color='blue', lw=2)
        ax.set_title('Rendements (%)', fontweight='bold')
        ax.set_ylabel('Rendements')
        ax.set_xlabel('Date')

    @staticmethod
    def plot_drawdown(drawdown, ax):
        ax.fill_between(drawdown.index, drawdown.values, color='red')
        ax.set_title('Drawdown (%)', fontweight='bold')
        ax.set_ylabel('Drawdown')
        ax.set_xlabel('Date')

    @staticmethod
    def create_performance_table(metrics_data, ax):
        ax.axis('off')
        metrics_df = pd.DataFrame(metrics_data,index=[0])
        table_data = [[col, metrics_df[col].values[0]] for col in metrics_df.columns]
        table = ax.table(cellText=table_data, colLabels=["Métrique", "Valeur"], cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(weight='bold')
    @staticmethod
    def calculate_monthly_returns(portfolio_values):
        monthly_returns = portfolio_values['Value'].resample('M').ffill().pct_change().dropna()
        return monthly_returns

    @staticmethod
    def prepare_heatmap_data(monthly_returns):
        # Supposons que monthly_returns est une Series avec un DateTimeIndex
        heatmap_data = monthly_returns.to_frame().pivot_table(
            index=monthly_returns.index.month,
            columns=monthly_returns.index.year,
            values='Value'
        ).fillna(0)
        # Convertir les numéros des mois en noms de mois
        heatmap_data.index = heatmap_data.index.map(lambda x: calendar.month_name[x])
        # Trier l'index pour avoir les mois dans l'ordre chronologique
        heatmap_data = heatmap_data.reindex(calendar.month_name[1:])
        return heatmap_data

    @staticmethod
    def plot_heatmap(heatmap_data, ax):
        if heatmap_data.empty:
            print("No data to plot heatmap.")
            return

        sns.heatmap(heatmap_data, ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
        ax.set_title('Rendements Mensuels (%)')
        ax.set_ylabel('Mois')
        ax.set_xlabel('Année')
        ax.set_yticklabels(heatmap_data.index, rotation=0)

    @staticmethod
    def stat_dashboard(portfolio_values, window, risk_free_rate=0):
        returns = PerformanceMetrics.calculate_log_returns(portfolio_values)
        cumulative_returns = PerformanceMetrics.cumulative_returns(portfolio_values)
        total_return = PerformanceMetrics.calculate_total_return(portfolio_values)
        annualized_return = PerformanceMetrics.calculate_annualized_return(portfolio_values)
        volatility = PerformanceMetrics.calculate_volatility(portfolio_values)
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(portfolio_values, risk_free_rate)
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(portfolio_values)
        sortino_ratio = PerformanceMetrics.calculate_sortino_ratio(portfolio_values, risk_free_rate)
        drawdown = PerformanceMetrics.calculate_drawdown(portfolio_values)
        monthly_returns = PerformanceMetrics.calculate_monthly_returns(portfolio_values)
        monthly_heatmap_data = PerformanceMetrics.prepare_heatmap_data(monthly_returns)

        fig, axs = plt.subplots(3, 2, figsize=(15, 15), gridspec_kw={'hspace': 0.4, 'height_ratios': [1, 1, 2]})
        fig.suptitle('Dashboard de Performance', y=0.98, fontsize=20, fontweight='bold')
        plt.subplots_adjust(wspace=0.25)

        PerformanceMetrics.plot_cumulative_returns(cumulative_returns, axs[0, 0])
        PerformanceMetrics.plot_returns(returns, axs[0, 1])
        PerformanceMetrics.plot_drawdown(drawdown, axs[1, 0])

        gs = fig.add_gridspec(3, 1, hspace=0.4, height_ratios=[1, 1, 2])
        ax_heatmap = fig.add_subplot(gs[2, :])
        PerformanceMetrics.plot_heatmap(monthly_heatmap_data, ax_heatmap)

        axs[2, 1].set_visible(False)
        axs[2, 0].set_visible(False)
        PerformanceMetrics.create_performance_table({
            "Rendement Total (%)": round(total_return * 100, 2),
            "Rendement Annualisé (%)": round(annualized_return * 100, 2),
            "Volatilité (%)": round(volatility * 100, 2),
            "Ratio de Sharpe": round(sharpe_ratio, 2),
            "Ratio de Sortino": round(sortino_ratio, 2),
            "Drawdown Max": round(max_drawdown, 2)
        }, axs[1, 1])

        plt.tight_layout()
        plt.show(block=False)
        window.mainloop()

class CryptoSelector:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Backtest Cryptocurrencies")

        self.date_frame = tk.Frame(self.window)
        self.crypto_frame = tk.Frame(self.window)
        self.strategy_frame = tk.Frame(self.window)
 
        self.date_frame.pack()
        self.crypto_frame.pack_forget()
        self.strategy_frame.pack_forget()

        self.create_date_widgets()
        self.previous_frame = None

    def create_date_widgets(self):
        self.start_date_label = tk.Label(self.date_frame, text="Start Date:")
        self.start_date_label.pack()
        self.start_date_entry = DateEntry(self.date_frame, date_pattern='y-mm-dd')
        self.start_date_entry.set_date(datetime.now() - timedelta(days=365))
        self.start_date_entry.pack()

        self.end_date_label = tk.Label(self.date_frame, text="End Date:")
        self.end_date_label.pack()
        self.end_date_entry = DateEntry(self.date_frame, date_pattern='y-mm-dd', maxdate=datetime.now() - timedelta(days=1))
        self.end_date_entry.pack()

        self.submit_button = ttk.Button(self.date_frame, text="Submit", command=self.submit)
        self.submit_button.pack()

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def submit(self):
        start_date = self.start_date_entry.get_date()
        end_date = self.end_date_entry.get_date()

        if start_date >= end_date:
            messagebox.showerror("Error", "Start date must be earlier than end date.")
            return

        self.start_date = start_date.strftime('%Y-%m-%d')
        self.end_date = end_date.strftime('%Y-%m-%d')
        print(f'From {self.start_date} to {self.end_date}')
        self.data_loader = DataLoader(self.start_date, self.end_date)
        symbols_list = self.data_loader.symbols_list

        # Set previous_frame before hiding the date frame
        self.previous_frame = self.date_frame
        self.date_frame.pack_forget()
        self.crypto_frame.pack()

        # Add a title to the cryptocurrency selection page
        self.crypto_title_label = tk.Label(self.crypto_frame, text="Select the cryptocurrencies for your strategy:")
        self.crypto_title_label.pack()

        self.crypto_listbox = tk.Listbox(self.crypto_frame, listvariable=tk.StringVar(value=symbols_list), selectmode="multiple")
        self.crypto_listbox.pack()

        self.select_button = ttk.Button(self.crypto_frame, text="Select", command=self.select)
        self.select_button.pack()

        self.back_button = ttk.Button(self.crypto_frame, text="Back", command=self.go_back)
        self.back_button.pack()

    def select(self):
        selected_indices = self.crypto_listbox.curselection()
        self.selected_cryptos = [self.crypto_listbox.get(i) for i in selected_indices]
        print("Selected Cryptocurrencies: ", self.selected_cryptos)

        self.previous_frame = self.crypto_frame
        self.crypto_frame.pack_forget()

        # Hide the crypto frame and show the strategy frame
        self.crypto_frame.pack_forget()
        self.create_strategy_widgets()
        self.previous_frame = self.crypto_frame
        self.crypto_frame.pack_forget()

    def go_back(self):
        if self.previous_frame is not None:
            if self.previous_frame == self.date_frame:
                self.crypto_frame.pack_forget()
            elif self.previous_frame == self.crypto_frame:
                self.strategy_frame.pack_forget()
            self.previous_frame.pack()

    def create_strategy_widgets(self):
        self.strategy_frame.pack()

        # Check if the strategy widgets already exist
        if not hasattr(self, 'strategy_label'):
            self.strategy_label = tk.Label(self.strategy_frame, text="Select Strategy:")
            self.strategy_var = tk.StringVar()
            self.strategy_var.set("EqualWeightStrategy")  # default value
            self.strategy_optionmenu = tk.OptionMenu(self.strategy_frame, self.strategy_var, "EqualWeightStrategy", "MarketCapStrategy")
            self.initial_capital_label = tk.Label(self.strategy_frame, text="Initial Capital:")
            self.initial_capital_entry = tk.Entry(self.strategy_frame)
            self.rebalancing_window_label = tk.Label(self.strategy_frame, text="Rebalancing Window:")
            self.rebalancing_window_entry = tk.Entry(self.strategy_frame)
            self.strategy_button = ttk.Button(self.strategy_frame, text="Submit", command=self.submit_strategy)
            self.back_button = ttk.Button(self.strategy_frame, text="Back", command=self.go_back)

        # Display the strategy widgets
        self.strategy_label.pack()
        self.strategy_optionmenu.pack()
        self.initial_capital_label.pack()
        self.initial_capital_entry.pack()
        self.rebalancing_window_label.pack()
        self.rebalancing_window_entry.pack()
        self.strategy_button.pack()
        self.back_button.pack()

    def submit_strategy(self):
        selected_strategy = self.strategy_var.get()
        initial_capital = float(self.initial_capital_entry.get())
        rebalancing_window = int(self.rebalancing_window_entry.get())
        print("Selected Strategy: ", selected_strategy)

        # Initialize the selected strategy
        if selected_strategy == "EqualWeightStrategy":
            strategy = EqualWeightStrategy(data=self.data_loader, rebalancing_window=rebalancing_window, initial_capital=initial_capital,
                                           start_date=self.start_date, end_date=self.end_date, selected_cryptos=self.selected_cryptos)
        elif selected_strategy == "MarketCapStrategy":
            strategy = MarketCapStrategy(data=self.data_loader, market_caps="path", rebalancing_window=rebalancing_window, initial_capital=initial_capital,
                                         start_date=self.start_date, end_date=self.end_date)

        # Run the strategy
        strategy.backtest.run()

        # Calculate returns and display performance dashboard
        returns = PerformanceMetrics.calculate_log_returns(strategy.portfolio_value)

        PerformanceMetrics.stat_dashboard(strategy.portfolio_value, self.window)

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':

    app = CryptoSelector()
    app.run()


    '''t1 = time.time()
    start_date = "2019-01-01"
    end_date = "2022-12-31"
    
    datas = DataLoader(binance_api_key, binance_api_secret, start_date, end_date)
    
    test = EqualWeightStrategy(datas, 30, 100000, start_date, end_date)
    r = PerformanceMetrics.calculate_returns(test.portfolio_value)
    PerformanceMetrics.stat_dashboard(test.portfolio_value, r, 1)
    t2 = time.time()
    minutes, seconds = divmod(t2 - t1, 60)
    print(f'Time: {int(minutes)} minutes {seconds} seconds')'''
