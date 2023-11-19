from data.coingecko_api import CoinGeckoData

if __name__ == "__main__":
    # Initialiser l'instance de la classe CoinGeckoData
    coingecko_data = CoinGeckoData()

    # Récupérer la liste des actifs côtés sur Binance avec leurs informations
    binance_assets_info = coingecko_data.get_binance_assets_info()

    if binance_assets_info:
        # Faire quelque chose avec les informations récupérées, par exemple les afficher
        for asset_info in binance_assets_info:
            print(asset_info)
    else:
        print("Impossible de récupérer les informations sur les actifs côtés sur Binance.")
