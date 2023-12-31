# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:27:57 2023

@author: paul-
"""

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