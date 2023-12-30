    Présentation de notre framework de backtesting d'indices de cryptomonnaies.

Le code est structuré selon les classes suivantes:<br>
1- La classe data loader pour récupérer les données.<br>
2- La classe 'AbstractStrategy' qui sera utile pour l'implémentation des classes de stratégie.<br>
3- Les classes 'MarketCapStrategy', 'EqualWeightStrategy' et 'PriceWeightedStrategy' qui correspondent aux trois types d'indices que l'utilisateur peut choisir de construire.<br>
4- La classe 'Backtest' qui exécute le processus de backtesting en itérant à travers les données historiques et en appliquant la stratégie.<br>
5- La classe 'PerformanceMetrics' qui renvoie les principaux indicateurs de performance d'une stratégie créee par l'utilisateur.<br>
6- La clase 'IndexCompositionTracker' qui renvoie la composition historique des actifs de la stratéie.<br>


    Classe DataLoader :
    
Elle est responsable de récupérer des données historiques de prix depuis Binance et des données de capitalisation boursière depuis CoinGecko.<br>
Ses différentes méthodes sont:<br>
- get_tickers : Récupère une liste de paires de trading (symboles) de Binance qui sont échangées contre USDT.<br>
- get_historical_market_caps : Récupère des données historiques de capitalisation boursière pour chaque cryptomonnaie dans la plage de dates spécifiée.<br>
- get_data_for_coin : Récupère des données historiques de prix pour une paire de trading donnée (symbole).<br>
- get_data : Récupère des données historiques de prix pour toutes les paires de trading.<br>
- combine_data : Combine les données historiques de prix pour toutes les paires de trading dans un seul DataFrame.<br>


    Classe AbstractStrategy :
    
Une classe de base abstraite définissant deux méthodes abstraites : calculate_weights et apply_strategy.<br>


    Classe MarketCapStrategy :
    
- Hérite de AbstractStrategy.
- Implémente des méthodes pour calculer des poids basés sur la capitalisation boursière et pour rééquilibrer le portefeuille en conséquence.
- Utilise la classe Backtest pour exécuter le processus de backtesting.
- Ses méthodes sont:<br>
    - __init__(self, data, market_caps, rebalancing_window, initial_capital, start_date, end_date) :
Initialise la stratégie avec les données nécessaires, les capitalisations boursières, la fenêtre de rééquilibrage, le capital initial, la date de début et la date de fin.<br>
    - calculate_weights(self) : Calcule les poids des actifs en fonction de leurs capitalisations boursières.
    - rebalancing(self) : Effectue le rééquilibrage du portefeuille en fonction des nouveaux poids calculés.
    - go_short(self, coin) : Effectue l'action de vente à découvert pour un actif donné.
    - go_long(self, coin) : Effectue l'action d'achat pour un actif donné.
    - apply_strategy(self) : Applique la stratégie en effectuant le rééquilibrage si nécessaire.<br>


    Classe EqualWeightStrategy :<br>
    
Elle est structurée de la même manière que la classe précedente. Les principales différences étant les inputs pris en compte (pas de market-cap ici) ainsi que dans la façon dont les poids des actifs sont calculés (tous équivalents).<br>


    Classe PriceWeightedStrategy :
  
Suis la même structure que la classe précedente hormi dans la façon dont les poids des actifs sont calculés (en fonction de leur prix).<br>


    Classe Backtest :
  
  Exécute le processus de backtesting en itérant à travers les données historiques et en appliquant la stratégie.<br>


    Classe PerformanceMetrics :
  
- Fournit des méthodes statiques pour calculer diverses métriques de performance telles que les rendements, les rendements cumulatifs, la volatilité, le ratio de Sharpe, le ratio de Sortino, le drawdown maximal et le rendement annualisé.
- Inclut une méthode pour afficher un tableau de bord de performance.


    Classe IndexCompositionTracker :
  
- Suit la composition de l'indice (portefeuille) à différents moments.
- Utilisation Exemplaire :
- Des instances de EqualWeightStrategy et PriceWeightedStrategy sont créées et testées en utilisant le chargeur de données fourni (DataLoader).
Les métriques de performance sont calculées et affichées en utilisant la classe PerformanceMetrics.
