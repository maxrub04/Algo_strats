"""
Strategy Correlation and Independence Analysis
Checks for correlation between strategies and autocorrelation in returns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
from statsmodels.stats.stattools import durbin_watson


class StrategyCorrelationAnalyzer:
    """
    Analyzes correlation between multiple trading strategies
    and tests for independence of returns
    """

    def __init__(self, strategies_data: Dict[str, pd.DataFrame]):
        """
        Args:
            strategies_data: Dictionary of {strategy_name: trades_df}
                            Each trades_df must have 'Profit' and 'Date' columns
        """
        self.strategies = strategies_data
        self.strategy_names = list(strategies_data.keys())

        # Align all strategies by date
        self.aligned_data = self._align_strategies()

    def _align_strategies(self) -> pd.DataFrame:
        """
        Align all strategies by date and create a unified DataFrame
        """
        aligned = {}

        for name, df in self.strategies.items():
            if 'Date' not in df.columns:
                raise ValueError(f"Strategy '{name}' missing 'Date' column")

            # Convert to datetime
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])

            # Set date as index and get profits
            df_indexed = df.set_index('Date')['Profit']
            df_indexed.name = name
            aligned[name] = df_indexed

        # Merge all strategies
        result = pd.DataFrame(aligned)

        # Fill NaN with 0 (days with no trade)
        result = result.fillna(0)

        return result

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate Pearson correlation matrix between strategies"""
        return self.aligned_data.corr()

    def calculate_spearman_correlation(self) -> pd.DataFrame:
        """Calculate Spearman rank correlation (non-parametric)"""
        return self.aligned_data.corr(method='spearman')

    def durbin_watson_test(self) -> Dict[str, float]:
        """
        Perform Durbin-Watson test on each strategy
        Tests for autocorrelation in returns

        DW statistic interpretation:
        - 2.0: No autocorrelation
        - < 2.0: Positive autocorrelation
        - > 2.0: Negative autocorrelation
        - Close to 2.0 indicates independence

        Returns:
            Dictionary of {strategy_name: dw_statistic}
        """
        dw_results = {}

        for name in self.strategy_names:
            returns = self.aligned_data[name].values
            # Remove zeros for more accurate test
            returns_nonzero = returns[returns != 0]

            if len(returns_nonzero) > 2:
                dw_stat = durbin_watson(returns_nonzero)
                dw_results[name] = dw_stat
            else:
                dw_results[name] = np.nan

        return dw_results

    def ljung_box_test(self, lags: int = 10) -> Dict[str, Dict]:
        """
        Perform Ljung-Box test for independence
        Tests whether returns are independently distributed

        Args:
            lags: Number of lags to test

        Returns:
            Dictionary with test statistics and p-values
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_results = {}

        for name in self.strategy_names:
            returns = self.aligned_data[name].values
            returns_nonzero = returns[returns != 0]

            if len(returns_nonzero) > lags:
                result = acorr_ljungbox(returns_nonzero, lags=lags, return_df=True)
                lb_results[name] = {
                    'lb_stat': result['lb_stat'].iloc[-1],
                    'p_value': result['lb_pvalue'].iloc[-1],
                    'is_independent': result['lb_pvalue'].iloc[-1] > 0.05  # 5% significance
                }
            else:
                lb_results[name] = {
                    'lb_stat': np.nan,
                    'p_value': np.nan,
                    'is_independent': None
                }

        return lb_results

    def runs_test(self) -> Dict[str, Dict]:
        """
        Perform Runs Test for randomness
        Tests if wins/losses are randomly distributed

        Returns:
            Dictionary with z-score and p-value for each strategy
        """
        runs_results = {}

        for name in self.strategy_names:
            returns = self.aligned_data[name].values
            returns_nonzero = returns[returns != 0]

            if len(returns_nonzero) < 10:
                runs_results[name] = {'z_score': np.nan, 'p_value': np.nan, 'is_random': None}
                continue

            # Convert to binary (win=1, loss=0)
            binary = (returns_nonzero > 0).astype(int)

            # Count runs
            runs = 1
            for i in range(1, len(binary)):
                if binary[i] != binary[i - 1]:
                    runs += 1

            # Calculate expected runs and variance
            n_wins = np.sum(binary)
            n_losses = len(binary) - n_wins

            if n_wins == 0 or n_losses == 0:
                runs_results[name] = {'z_score': np.nan, 'p_value': np.nan, 'is_random': None}
                continue

            expected_runs = (2 * n_wins * n_losses) / len(binary) + 1
            variance = (2 * n_wins * n_losses * (2 * n_wins * n_losses - len(binary))) / \
                       (len(binary) ** 2 * (len(binary) - 1))

            # Z-score
            z_score = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            runs_results[name] = {
                'z_score': z_score,
                'p_value': p_value,
                'is_random': p_value > 0.05  # 5% significance
            }

        return runs_results

    def calculate_diversification_ratio(self) -> float:
        """
        Calculate portfolio diversification ratio
        Higher value indicates better diversification

        Returns:
            Diversification ratio
        """
        # Weighted average volatility / Portfolio volatility
        volatilities = self.aligned_data.std()
        avg_volatility = volatilities.mean()

        # Portfolio volatility (considering correlations)
        portfolio_variance = self.aligned_data.sum(axis=1).var()
        portfolio_vol = np.sqrt(portfolio_variance)

        if portfolio_vol == 0:
            return 0

        div_ratio = (avg_volatility * len(self.strategy_names)) / portfolio_vol

        return div_ratio

    def print_correlation_report(self):
        """Print comprehensive correlation analysis report"""
        print("\n" + "=" * 80)
        print("STRATEGY CORRELATION & INDEPENDENCE ANALYSIS")
        print("=" * 80)

        # 1. Correlation Matrix
        print("\n1. PEARSON CORRELATION MATRIX")
        print("-" * 80)
        corr_matrix = self.calculate_correlation_matrix()
        print(corr_matrix.round(3).to_string())

        # 2. Spearman Correlation
        print("\n2. SPEARMAN RANK CORRELATION")
        print("-" * 80)
        spearman = self.calculate_spearman_correlation()
        print(spearman.round(3).to_string())

        # 3. Durbin-Watson Test
        print("\n3. DURBIN-WATSON TEST (Autocorrelation)")
        print("-" * 80)
        print("Interpretation: ~2.0 = no autocorrelation, <2.0 = positive, >2.0 = negative")
        dw_results = self.durbin_watson_test()
        for name, dw in dw_results.items():
            status = "Independent" if 1.5 < dw < 2.5 else "**CORRELATED**"
            print(f"  {name:.<40} {dw:.4f}  [{status}]")

        # 4. Ljung-Box Test
        print("\n4. LJUNG-BOX TEST (Independence)")
        print("-" * 80)
        print("H0: Returns are independently distributed (p > 0.05 = independent)")
        lb_results = self.ljung_box_test()
        for name, result in lb_results.items():
            if result['is_independent'] is not None:
                status = "✓ Independent" if result['is_independent'] else "✗ NOT Independent"
                print(f"  {name:.<40} p={result['p_value']:.4f}  [{status}]")

        # 5. Runs Test
        print("\n5. RUNS TEST (Randomness)")
        print("-" * 80)
        print("H0: Win/Loss sequence is random (p > 0.05 = random)")
        runs_results = self.runs_test()
        for name, result in runs_results.items():
            if result['is_random'] is not None:
                status = "✓ Random" if result['is_random'] else "✗ NOT Random"
                print(f"  {name:.<40} p={result['p_value']:.4f}  [{status}]")

        # 6. Diversification
        print("\n6. PORTFOLIO DIVERSIFICATION")
        print("-" * 80)
        div_ratio = self.calculate_diversification_ratio()
        print(f"  Diversification Ratio: {div_ratio:.3f}")
        print(f"  (>1.0 indicates diversification benefit)")

        print("\n" + "=" * 80)

    def plot_correlation_analysis(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of correlation analysis
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Correlation Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        corr_matrix = self.calculate_correlation_matrix()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1,
                    vmin=-1, vmax=1)
        ax1.set_title('Strategy Correlation Matrix (Pearson)', fontsize=14, fontweight='bold')

        # 2. Durbin-Watson Results
        ax2 = fig.add_subplot(gs[0, 2])
        dw_results = self.durbin_watson_test()
        names = list(dw_results.keys())
        values = list(dw_results.values())
        colors = ['green' if 1.5 < v < 2.5 else 'red' for v in values]

        ax2.barh(names, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(2.0, color='blue', linestyle='--', linewidth=2, label='Perfect Independence')
        ax2.axvline(1.5, color='orange', linestyle=':', linewidth=1)
        ax2.axvline(2.5, color='orange', linestyle=':', linewidth=1)
        ax2.set_xlabel('Durbin-Watson Statistic')
        ax2.set_title('Durbin-Watson Test', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. Cumulative Returns
        ax3 = fig.add_subplot(gs[1, :])
        cumulative = self.aligned_data.cumsum()
        for col in cumulative.columns:
            ax3.plot(cumulative.index, cumulative[col], label=col, linewidth=2)
        ax3.set_title('Cumulative Returns by Strategy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Profit ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Rolling Correlation (if multiple strategies)
        if len(self.strategy_names) >= 2:
            ax4 = fig.add_subplot(gs[2, 0])
            # Calculate rolling correlation between first two strategies
            rolling_corr = self.aligned_data[self.strategy_names[0]].rolling(30).corr(
                self.aligned_data[self.strategy_names[1]]
            )
            ax4.plot(rolling_corr.index, rolling_corr, color='purple', linewidth=2)
            ax4.axhline(0, color='black', linestyle='--', linewidth=1)
            ax4.set_title(f'Rolling Correlation (30-day)\n{self.strategy_names[0]} vs {self.strategy_names[1]}',
                          fontsize=12, fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Correlation')
            ax4.grid(True, alpha=0.3)

        # 5. Scatter Matrix (if 2-3 strategies)
        if 2 <= len(self.strategy_names) <= 3:
            ax5 = fig.add_subplot(gs[2, 1])
            if len(self.strategy_names) == 2:
                ax5.scatter(self.aligned_data[self.strategy_names[0]],
                            self.aligned_data[self.strategy_names[1]],
                            alpha=0.5, s=20)
                ax5.set_xlabel(self.strategy_names[0])
                ax5.set_ylabel(self.strategy_names[1])
                ax5.set_title('Strategy Returns Scatter', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
            else:
                from mpl_toolkits.mplot3d import Axes3D
                ax5 = fig.add_subplot(gs[2, 1], projection='3d')
                ax5.scatter(self.aligned_data[self.strategy_names[0]],
                            self.aligned_data[self.strategy_names[1]],
                            self.aligned_data[self.strategy_names[2]],
                            alpha=0.5, s=20)
                ax5.set_xlabel(self.strategy_names[0])
                ax5.set_ylabel(self.strategy_names[1])
                ax5.set_zlabel(self.strategy_names[2])
                ax5.set_title('3D Returns Scatter', fontsize=12, fontweight='bold')

        # 6. Independence Test Summary
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')

        # Create summary text
        summary_text = "INDEPENDENCE TESTS\n" + "=" * 30 + "\n\n"

        dw_results = self.durbin_watson_test()
        lb_results = self.ljung_box_test()
        runs_results = self.runs_test()

        for name in self.strategy_names:
            summary_text += f"{name}:\n"

            if name in dw_results:
                dw = dw_results[name]
                dw_status = "✓" if 1.5 < dw < 2.5 else "✗"
                summary_text += f"  DW: {dw:.3f} {dw_status}\n"

            if name in lb_results and lb_results[name]['is_independent'] is not None:
                lb_status = "✓" if lb_results[name]['is_independent'] else "✗"
                summary_text += f"  LB: p={lb_results[name]['p_value']:.3f} {lb_status}\n"

            if name in runs_results and runs_results[name]['is_random'] is not None:
                runs_status = "✓" if runs_results[name]['is_random'] else "✗"
                summary_text += f"  Runs: p={runs_results[name]['p_value']:.3f} {runs_status}\n"

            summary_text += "\n"

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Strategy Correlation & Independence Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()


def analyze_correlation_from_files(filepaths: Dict[str, str],
                                   plot: bool = True,
                                   save_plot: Optional[str] = None) -> StrategyCorrelationAnalyzer:
    """
    Analyze correlation between multiple strategies from files

    Args:
        filepaths: Dictionary of {strategy_name: filepath}
        plot: Whether to create visualizations
        save_plot: Path to save plot (optional)

    Returns:
        StrategyCorrelationAnalyzer object
    """
    strategies_data = {}

    for name, filepath in filepaths.items():
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"File {filepath} must be .csv or .json")

        if 'Profit' not in df.columns:
            raise ValueError(f"Strategy '{name}' file must have 'Profit' column")

        strategies_data[name] = df

    # Create analyzer
    analyzer = StrategyCorrelationAnalyzer(strategies_data)

    # Print report
    analyzer.print_correlation_report()

    # Plot if requested
    if plot:
        analyzer.plot_correlation_analysis(save_path=save_plot)

    return analyzer


if __name__ == "__main__":
    print("Strategy Correlation & Independence Analyzer")
    print("=" * 70)
    print("\nUsage:")
    print("  from correlation_analyzer import analyze_correlation_from_files")
    print("  filepaths = {")
    print("      'Strategy A': 'strategy_a_trades.csv',")
    print("      'Strategy B': 'strategy_b_trades.csv'")
    print("  }")
    print("  analyzer = analyze_correlation_from_files(filepaths)")