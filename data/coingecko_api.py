# Importer le module pycoingecko
from pycoingecko import CoinGeckoAPI

class CoinGeckoData:
    def __init__(self):
        self.coingecko = CoinGeckoAPI()

    def get_binance_assets_info(self):
        # Récupérer les identifiants des actifs côtés sur binance
        ids = self.get_binance_ids()

        # Obtenir les données de market cap et de tag pour les actifs filtrés
        # Utiliser la devise USD comme référence
        vs_currency = "usd"
        binance_coins_markets = self.get_binance_markets(vs_currency, ids)

        # Créer une liste de dictionnaires contenant le market cap et le tag pour chaque actif
        assets_info = self.get_binance_info(binance_coins_markets)

        return assets_info

    def get_binance_ids(self):
        # Initialiser le numéro de page à 1
        page = 1

        # Créer une liste vide pour stocker les identifiants des actifs
        ids = []

        # Créer une variable booléenne pour indiquer si on a atteint la dernière page
        last_page = False

        # Tant qu'on n'a pas atteint la dernière page
        while not last_page:
            try:
                # Récupérer les tickers de la page courante
                page_tickers = self.coingecko.get_exchanges_tickers_by_id("binance", page=page)

                # Vérifier si la liste des tickers est vide
                if not page_tickers["tickers"]:
                    # Si oui, on a atteint la dernière page
                    last_page = True
                else:
                    # Sinon, on ajoute les identifiants des actifs à la liste globale
                    for ticker in page_tickers["tickers"]:
                        ids.append(ticker["coin_id"])

                    # On incrémente le numéro de page
                    page += 1

            except Exception as e:
                # Si une erreur se produit, on affiche le message d'erreur et on arrête la boucle
                print(f"Une erreur s'est produite lors de la récupération des tickers depuis CoinGecko: {e}")
                last_page = True

        return ids

    def get_binance_markets(self, vs_currency, ids):
        # Créer une liste vide pour stocker les données de marché
        binance_coins_markets = []

        # Définir la taille maximale des sous-listes d'identifiants
        max_size = 50

        # Diviser la liste des identifiants en sous-listes de taille maximale
        sublists = [ids[i:i+max_size] for i in range(0, len(ids), max_size)]

        # Pour chaque sous-liste d'identifiants
        for sublist in sublists:
            try:
                # Obtenir les données de marché avec la méthode get_coins_markets()
                sublist_markets = self.coingecko.get_coins_markets(vs_currency=vs_currency, ids=sublist)

                # Ajouter les données de marché à la liste globale
                binance_coins_markets.extend(sublist_markets)

            except Exception as e:
                # Si une erreur se produit, on affiche le message d'erreur et on arrête la boucle
                print(f"Une erreur s'est produite lors de la récupération des données de marché depuis CoinGecko: {e}")
                break

        return binance_coins_markets

    def get_binance_info(self, binance_coins_markets):
        # Créer une liste de dictionnaires contenant le market cap et le tag pour chaque actif
        assets_info = []
        for coin in binance_coins_markets:
            assets_info.append({
                'symbol': coin['symbol'],
                'market_cap': coin['market_cap'],
            })
    # Peut-être stocker les datas dans un Excel pour éviter de rappeler ce code à chaque fois (ce qui est un peu long) ?
        return assets_info

