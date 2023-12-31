from data.coingecko_api import CoinGeckoData
import pandas as pd

if __name__ == "__main__":

    # Créer une instance de la classe CoinGeckoData
    cg_data = CoinGeckoData()
    # Obtenir les informations des actifs côtés sur Binance
    assets_info = cg_data.get_binance_assets_info()
    # Créer un dataframe à partir de la liste de dictionnaires
    df = pd.DataFrame(assets_info)
    print(df)
    # Exporter le dataframe dans un fichier Excel
    #df.to_excel("binance_assets_info.xlsx")
