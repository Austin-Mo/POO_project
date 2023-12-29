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
Ces différentes méthodes sont:<br>
- get_tickers : Récupère une liste de paires de trading (symboles) de Binance qui sont échangées contre USDT.<br>
- get_historical_market_caps : Récupère des données historiques de capitalisation boursière pour chaque cryptomonnaie dans la plage de dates spécifiée.<br>
- get_data_for_coin : Récupère des données historiques de prix pour une paire de trading donnée (symbole).<br>
- get_data : Récupère des données historiques de prix pour toutes les paires de trading.<br>
- combine_data : Combine les données historiques de prix pour toutes les paires de trading dans un seul DataFrame.<br>


    Classe AbstractStrategy :
    
Une classe de base abstraite définissant deux méthodes abstraites : calculate_weights et apply_strategy.


    Classes MarketCapStrategy :
    
- Hérite de AbstractStrategy.
- Implémente des méthodes pour calculer des poids basés sur la capitalisation boursière et pour rééquilibrer le portefeuille en conséquence.
- Utilise la classe Backtest pour exécuter le processus de backtesting.


    Classe EqualWeightStrategy :
  
- Hérite de AbstractStrategy.
- Implémente des méthodes pour calculer des poids égaux pour chaque paire de trading et pour rééquilibrer le portefeuille en conséquence.
- Utilise la classe Backtest pour exécuter le processus de backtesting.


    Classe PriceWeightedStrategy :
  
- Hérite de AbstractStrategy.
- Implémente des méthodes pour calculer des poids basés sur les prix de clôture et pour rééquilibrer le portefeuille en conséquence.
- Utilise la classe Backtest pour exécuter le processus de backtesting.


    Classe Backtest :
  
- Exécute le processus de backtesting en itérant à travers les données historiques et en appliquant la stratégie.


    Classe PerformanceMetrics :
  
- Fournit des méthodes statiques pour calculer diverses métriques de performance telles que les rendements, les rendements cumulatifs, la volatilité, le ratio de Sharpe, le ratio de Sortino, le drawdown maximal et le rendement annualisé.
- Inclut une méthode pour afficher un tableau de bord de performance.


    Classe IndexCompositionTracker :
  
-Suit la composition de l'indice (portefeuille) à différents moments.
-Utilisation Exemplaire :
-Des instances de EqualWeightStrategy et PriceWeightedStrategy sont créées et testées en utilisant le chargeur de données fourni (DataLoader).
Les métriques de performance sont calculées et affichées en utilisant la classe PerformanceMetrics.
