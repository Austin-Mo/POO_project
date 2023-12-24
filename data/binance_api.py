from binance.client import Client

class BinanceData:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)

    def get_historical_prices(self, symbol, start_date, end_date):
        # Récupérer les prix historiques depuis l'API Binance sans clé API (limité aux 1000 dernières bougies)
        klines = self.client.get_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1DAY,
            limit=1000
        )

        # Extraire les prix de clôture (Close) de chaque bougie
        closing_prices = [float(kline[4]) for kline in klines]

        return closing_prices
