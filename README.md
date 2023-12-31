**Présentation de notre framework de backtesting d'indices de cryptomonnaies**

Le code est structuré selon les classes suivantes:<br>
1- La classe data loader pour récupérer les données.<br>
2- La classe 'AbstractStrategy' qui sera utile pour l'implémentation des classes de stratégie.<br>
3- Les classes 'MarketCapStrategy', 'EqualWeightStrategy' et 'PriceWeightedStrategy' qui correspondent aux trois types d'indices que l'utilisateur peut choisir de construire.<br>
4- La classe 'Backtest' qui exécute le processus de backtesting en itérant à travers les données historiques et en appliquant la stratégie.<br>
5- La classe 'PerformanceMetrics' qui renvoie les principaux indicateurs de performance d'une stratégie créee par l'utilisateur.<br>
6- La clase 'IndexCompositionTracker' qui renvoie la composition historique des actifs de la stratégie.<br>


**Classe DataLoader**
    
Elle est responsable de récupérer des données historiques de prix depuis Binance et des données de capitalisation boursière depuis CoinGecko.<br>
Ses différentes méthodes sont:<br>
- get_tickers : Récupère une liste de paires de trading (symboles) de Binance qui sont échangées contre USDT.<br>
- get_historical_market_caps : Récupère des données historiques de capitalisation boursière pour chaque cryptomonnaie dans la plage de dates spécifiée.<br>
- get_data_for_coin : Récupère des données historiques de prix pour une paire de trading donnée (symbole).<br>
- get_data : Récupère des données historiques de prix pour toutes les paires de trading.<br>
- combine_data : Combine les données historiques de prix pour toutes les paires de trading dans un seul DataFrame.<br>


**Classe AbstractStrategy**
    
Une classe de base abstraite définissant deux méthodes abstraites : calculate_weights et apply_strategy.<br>


**Classe MarketCapStrategy**
    
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


**Classe EqualWeightStrategy**
    
Elle est structurée de la même manière que la classe précedente. La principale différence étant la manière dont les poids ainsi que les allocations initiales de chaque actif sont attribués.


**Classe PriceWeightedStrategy**
  
Ici encore, la classe suit la même structure que la classe précedente hormi dans la façon dont les poids des actifs sont calculés (en fonction de leur prix).<br>


**Classe Backtest**
  
Exécute le processus de backtesting en itérant à travers les données historiques et en appliquant la stratégie.<br>


**Classe PerformanceMetrics**
  
- Fournit des méthodes statiques pour calculer diverses métriques de performance telles que les rendements, les rendements cumulatifs, la volatilité, le ratio de Sharpe, le ratio de Sortino, le drawdown maximal et le rendement annualisé.
- Inclut une méthode pour afficher un tableau de bord de performance.


**Classe IndexCompositionTracker**
  
- Suit la composition de l'indice (portefeuille) à différents moments.



**Utilisation Exemplaire par l'utilisateur**

- L'utilisateur récupère des données en séléctionnant une date de début, une date de fin ainsi que ses clés API Binance afin d'exécuter la classe DataLoader.<br>
- Ensuite, l'utilisateur exécute la classe correspondant à la stratégie qu'il souhaite mettre en place en y intégrant les données récupérées dans la classe DataLoader puis en choissant une fenêtre de rebalancement, un capital de départ et les dates d'arrivée et de sortie.<br>
- Enfin, afin d'afficher le dashboard de performance, l'utilisateur devra d'abord exécuter la méthode 'calculate returns' de la classe PerformanceMetrics puis la méthode 'stat_dashboard' de la même classe en y ajoutant les valeurs de portefeuille de la stratégie ainsi que ses rendements journaliers.
