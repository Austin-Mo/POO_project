# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:26:04 2023

@author: paul-
"""

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
    def stat_dashboard(portfolio_values, risk_free_rate=0):
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