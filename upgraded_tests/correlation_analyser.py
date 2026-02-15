"""
Strategy Correlation and Independence Analysis
Checks for correlation between strategies and autocorrelation in returns
v2: Added Combined Portfolio Equity Curve panel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
from statsmodels.stats.stattools import durbin_watson


class StrategyCorrelationAnalyzer:
    """
    Analyzes correlation between multiple trading strategies
    and tests for independence of returns
    """

    def __init__(self, strategies_data: Dict[str, pd.DataFrame], initial_capital: float = 10000.0):
        """
        Args:
            strategies_data: Dictionary of {strategy_name: trades_df}
                            Each trades_df must have 'Profit' and 'Date' columns
            initial_capital: Starting capital per strategy (used for combined equity curve)
        """
        self.strategies = strategies_data
        self.strategy_names = list(strategies_data.keys())
        self.initial_capital = initial_capital

        # Align all strategies by date
        self.aligned_data = self._align_strategies()

    def _align_strategies(self) -> pd.DataFrame:
        """Align all strategies by date and create a unified DataFrame"""
        aligned = {}

        for name, df in self.strategies.items():
            if 'Date' not in df.columns:
                raise ValueError(f"Strategy '{name}' missing 'Date' column")

            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df_indexed = df.set_index('Date')['Profit']
            df_indexed.name = name
            aligned[name] = df_indexed

        result = pd.DataFrame(aligned)
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
        DW ~2.0 = no autocorrelation | <2.0 = positive | >2.0 = negative
        """
        dw_results = {}
        for name in self.strategy_names:
            returns = self.aligned_data[name].values
            returns_nonzero = returns[returns != 0]
            if len(returns_nonzero) > 2:
                dw_stat = durbin_watson(returns_nonzero)
                dw_results[name] = dw_stat
            else:
                dw_results[name] = np.nan
        return dw_results

    def ljung_box_test(self, lags: int = 10) -> Dict[str, Dict]:
        """Perform Ljung-Box test for independence"""
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
                    'is_independent': result['lb_pvalue'].iloc[-1] > 0.05
                }
            else:
                lb_results[name] = {'lb_stat': np.nan, 'p_value': np.nan, 'is_independent': None}
        return lb_results

    def runs_test(self) -> Dict[str, Dict]:
        """Perform Runs Test for randomness of win/loss sequence"""
        runs_results = {}
        for name in self.strategy_names:
            returns = self.aligned_data[name].values
            returns_nonzero = returns[returns != 0]
            if len(returns_nonzero) < 10:
                runs_results[name] = {'z_score': np.nan, 'p_value': np.nan, 'is_random': None}
                continue

            binary = (returns_nonzero > 0).astype(int)
            runs = 1
            for i in range(1, len(binary)):
                if binary[i] != binary[i - 1]:
                    runs += 1

            n_wins = np.sum(binary)
            n_losses = len(binary) - n_wins

            if n_wins == 0 or n_losses == 0:
                runs_results[name] = {'z_score': np.nan, 'p_value': np.nan, 'is_random': None}
                continue

            expected_runs = (2 * n_wins * n_losses) / len(binary) + 1
            variance = (2 * n_wins * n_losses * (2 * n_wins * n_losses - len(binary))) / \
                       (len(binary) ** 2 * (len(binary) - 1))

            z_score = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            runs_results[name] = {'z_score': z_score, 'p_value': p_value, 'is_random': p_value > 0.05}
        return runs_results

    def calculate_diversification_ratio(self) -> float:
        """Calculate portfolio diversification ratio. Higher = better diversification."""
        volatilities = self.aligned_data.std()
        avg_volatility = volatilities.mean()
        portfolio_variance = self.aligned_data.sum(axis=1).var()
        portfolio_vol = np.sqrt(portfolio_variance)
        if portfolio_vol == 0:
            return 0
        return (avg_volatility * len(self.strategy_names)) / portfolio_vol

    def build_combined_equity_curve(self) -> pd.DataFrame:
        """
        Builds a combined portfolio equity curve by summing daily P&L
        across all strategies and applying to a single capital base.

        Each strategy contributes its trade P&L on each date.
        The combined curve treats all strategies as running simultaneously
        from the same initial capital.
        """
        # Sum daily profits across all strategies
        combined_daily_pnl = self.aligned_data.sum(axis=1)

        # Individual equity curves (capital + cumulative PnL)
        indiv_curves = pd.DataFrame(index=self.aligned_data.index)
        for name in self.strategy_names:
            indiv_curves[name] = self.initial_capital + self.aligned_data[name].cumsum()

        # Combined equity curve
        combined_equity = self.initial_capital + combined_daily_pnl.cumsum()

        return indiv_curves, combined_equity

    def print_correlation_report(self):
        """Print comprehensive correlation analysis report"""
        print("\n" + "=" * 80)
        print("STRATEGY CORRELATION & INDEPENDENCE ANALYSIS")
        print("=" * 80)

        print("\n1. PEARSON CORRELATION MATRIX")
        print("-" * 80)
        corr_matrix = self.calculate_correlation_matrix()
        print(corr_matrix.round(3).to_string())

        print("\n2. SPEARMAN RANK CORRELATION")
        print("-" * 80)
        spearman = self.calculate_spearman_correlation()
        print(spearman.round(3).to_string())

        print("\n3. DURBIN-WATSON TEST (Autocorrelation)")
        print("-" * 80)
        print("Interpretation: ~2.0 = no autocorrelation, <2.0 = positive, >2.0 = negative")
        dw_results = self.durbin_watson_test()
        for name, dw in dw_results.items():
            status = "Independent" if 1.5 < dw < 2.5 else "**CORRELATED**"
            print(f"  {name:.<40} {dw:.4f}  [{status}]")

        print("\n4. LJUNG-BOX TEST (Independence)")
        print("-" * 80)
        print("H0: Returns are independently distributed (p > 0.05 = independent)")
        lb_results = self.ljung_box_test()
        for name, result in lb_results.items():
            if result['is_independent'] is not None:
                status = "✓ Independent" if result['is_independent'] else "✗ NOT Independent"
                print(f"  {name:.<40} p={result['p_value']:.4f}  [{status}]")

        print("\n5. RUNS TEST (Randomness)")
        print("-" * 80)
        print("H0: Win/Loss sequence is random (p > 0.05 = random)")
        runs_results = self.runs_test()
        for name, result in runs_results.items():
            if result['is_random'] is not None:
                status = "✓ Random" if result['is_random'] else "✗ NOT Random"
                print(f"  {name:.<40} p={result['p_value']:.4f}  [{status}]")

        print("\n6. PORTFOLIO DIVERSIFICATION")
        print("-" * 80)
        div_ratio = self.calculate_diversification_ratio()
        print(f"  Diversification Ratio: {div_ratio:.3f}")
        print(f"  (>1.0 indicates diversification benefit)")
        print("\n" + "=" * 80)

    def plot_correlation_analysis(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization with combined equity curve.
        Layout:
          Row 0: Corr Heatmap | DW Chart
          Row 1: Individual + Combined Equity Curves (full width)  <-- NEW
          Row 2: Rolling Corr | MFE/MAE Scatter | Independence Summary
        """
        # ── Build data ──────────────────────────────────────────────────────────
        indiv_curves, combined_equity = self.build_combined_equity_curve()
        dw_results = self.durbin_watson_test()
        lb_results = self.ljung_box_test()
        runs_results = self.runs_test()

        # Max drawdown for combined
        running_max = combined_equity.cummax()
        drawdown_combined = (combined_equity - running_max) / running_max * 100

        # ── Layout ──────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(20, 18))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.40, wspace=0.30)

        # ── Panel 0-left: Pearson Heatmap ────────────────────────────────────
        ax_heat = fig.add_subplot(gs[0, :2])
        corr_matrix = self.calculate_correlation_matrix()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax_heat,
                    vmin=-1, vmax=1)
        ax_heat.set_title('Strategy Correlation Matrix (Pearson)', fontsize=13, fontweight='bold')

        # ── Panel 0-right: Durbin-Watson ─────────────────────────────────────
        ax_dw = fig.add_subplot(gs[0, 2])
        names = list(dw_results.keys())
        values = [v for v in dw_results.values()]
        colors_dw = ['#2ecc71' if 1.5 < v < 2.5 else '#e74c3c' for v in values]
        ax_dw.barh(names, values, color=colors_dw, alpha=0.85, edgecolor='black')
        ax_dw.axvline(2.0, color='royalblue', linestyle='--', linewidth=2, label='Perfect (2.0)')
        ax_dw.axvline(1.5, color='orange', linestyle=':', linewidth=1)
        ax_dw.axvline(2.5, color='orange', linestyle=':', linewidth=1)
        ax_dw.set_xlabel('Durbin-Watson Statistic')
        ax_dw.set_title('Durbin-Watson Test', fontsize=12, fontweight='bold')
        ax_dw.legend(fontsize=9)
        ax_dw.grid(True, alpha=0.3, axis='x')

        # ── Panel 1: Individual Equity Curves (top sub-panel) ────────────────
        ax_indiv = fig.add_subplot(gs[1, :])
        palette = plt.cm.tab10.colors
        for idx, name in enumerate(self.strategy_names):
            ax_indiv.plot(indiv_curves.index, indiv_curves[name],
                          label=name, linewidth=1.8,
                          color=palette[idx % len(palette)], alpha=0.75)
        ax_indiv.axhline(self.initial_capital, color='gray', linestyle=':', linewidth=1)
        ax_indiv.set_title('Individual Strategy Equity Curves', fontsize=13, fontweight='bold')
        ax_indiv.set_ylabel('Equity ($)')
        ax_indiv.legend(fontsize=10)
        ax_indiv.grid(True, alpha=0.25)

        # ── Panel 2: COMBINED Equity Curve (full width) ───────────────────────
        ax_comb = fig.add_subplot(gs[2, :])

        # Shade the area under/above initial capital
        ax_comb.fill_between(combined_equity.index, combined_equity,
                             self.initial_capital,
                             where=(combined_equity >= self.initial_capital),
                             interpolate=True, color='#2ecc71', alpha=0.20)
        ax_comb.fill_between(combined_equity.index, combined_equity,
                             self.initial_capital,
                             where=(combined_equity < self.initial_capital),
                             interpolate=True, color='#e74c3c', alpha=0.20)
        ax_comb.plot(combined_equity.index, combined_equity,
                     color='#2c3e50', linewidth=2.2, label='Combined Portfolio')
        ax_comb.axhline(self.initial_capital, color='gray', linestyle='--',
                        linewidth=1.2, label=f'Starting Capital (${self.initial_capital:,.0f})')

        # Annotate final value
        final_val = combined_equity.iloc[-1]
        total_ret = (final_val - self.initial_capital) / self.initial_capital * 100
        max_dd = drawdown_combined.min()
        ax_comb.annotate(
            f'Final: ${final_val:,.0f}  ({total_ret:+.1f}%)\nMax DD: {max_dd:.1f}%',
            xy=(combined_equity.index[-1], final_val),
            xytext=(-160, -40), textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'),
            arrowprops=dict(arrowstyle='->', color='gray')
        )

        ax_comb.set_title(
            f'Combined Portfolio Equity Curve  '
            f'({len(self.strategy_names)} strategies, same capital base)',
            fontsize=13, fontweight='bold'
        )
        ax_comb.set_ylabel('Equity ($)')
        ax_comb.set_xlabel('Date')
        ax_comb.legend(fontsize=10)
        ax_comb.grid(True, alpha=0.25)

        # ── Panel 3-left: Rolling Correlation ────────────────────────────────
        ax_roll = fig.add_subplot(gs[3, 0])
        if len(self.strategy_names) >= 2:
            rolling_corr = self.aligned_data[self.strategy_names[0]].rolling(30).corr(
                self.aligned_data[self.strategy_names[1]]
            )
            ax_roll.plot(rolling_corr.index, rolling_corr, color='purple', linewidth=1.8)
            ax_roll.axhline(0, color='black', linestyle='--', linewidth=1)
            ax_roll.fill_between(rolling_corr.index, rolling_corr, 0,
                                 where=(rolling_corr > 0), color='red', alpha=0.15)
            ax_roll.fill_between(rolling_corr.index, rolling_corr, 0,
                                 where=(rolling_corr <= 0), color='green', alpha=0.15)
            ax_roll.set_title(
                f'Rolling Corr (30-day)\n{self.strategy_names[0]} vs {self.strategy_names[1]}',
                fontsize=11, fontweight='bold'
            )
            ax_roll.set_xlabel('Date')
            ax_roll.set_ylabel('Correlation')
            ax_roll.set_ylim(-1, 1)
            ax_roll.grid(True, alpha=0.3)

        # ── Panel 3-mid: Scatter Plot ─────────────────────────────────────────
        ax_scat = fig.add_subplot(gs[3, 1])
        if len(self.strategy_names) == 2:
            ax_scat.scatter(
                self.aligned_data[self.strategy_names[0]],
                self.aligned_data[self.strategy_names[1]],
                alpha=0.45, s=20, color='steelblue', edgecolors='none'
            )
            ax_scat.set_xlabel(self.strategy_names[0])
            ax_scat.set_ylabel(self.strategy_names[1])
            ax_scat.set_title('Returns Scatter', fontsize=12, fontweight='bold')
            ax_scat.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_scat.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_scat.grid(True, alpha=0.3)

        # ── Panel 3-right: Independence Test Summary ──────────────────────────
        ax_sum = fig.add_subplot(gs[3, 2])
        ax_sum.axis('off')

        summary_text = "INDEPENDENCE TESTS\n" + "=" * 30 + "\n\n"
        for name in self.strategy_names:
            summary_text += f"{name}:\n"
            if name in dw_results:
                dw = dw_results[name]
                dw_ok = "✓" if 1.5 < dw < 2.5 else "✗"
                summary_text += f"  DW: {dw:.3f} {dw_ok}\n"
            if name in lb_results and lb_results[name]['is_independent'] is not None:
                lb_ok = "✓" if lb_results[name]['is_independent'] else "✗"
                summary_text += f"  LB: p={lb_results[name]['p_value']:.3f} {lb_ok}\n"
            if name in runs_results and runs_results[name]['is_random'] is not None:
                r_ok = "✓" if runs_results[name]['is_random'] else "✗"
                summary_text += f"  Runs: p={runs_results[name]['p_value']:.3f} {r_ok}\n"
            summary_text += "\n"

        div = self.calculate_diversification_ratio()
        summary_text += f"Div. Ratio: {div:.3f}\n"
        summary_text += f"({'✓ Good' if div > 1.0 else '✗ Low'} diversification)"

        ax_sum.text(0.05, 0.95, summary_text, transform=ax_sum.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35))

        plt.suptitle('Strategy Correlation & Independence Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.tight_layout()
        plt.show()


# ── Convenience function ──────────────────────────────────────────────────────

def analyze_correlation_from_files(filepaths: Dict[str, str],
                                   initial_capital: float = 10000.0,
                                   plot: bool = True,
                                   save_plot: Optional[str] = None):
    """
    Analyze correlation between multiple strategies from CSV/JSON files.

    Args:
        filepaths: {strategy_name: filepath}
        initial_capital: Starting capital for equity curve calculation
        plot: Whether to show visualizations
        save_plot: Optional path to save the figure
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

    analyzer = StrategyCorrelationAnalyzer(strategies_data, initial_capital=initial_capital)
    analyzer.print_correlation_report()

    if plot:
        analyzer.plot_correlation_analysis(save_path=save_plot)

    return analyzer


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filepaths = {
        "IB H1": "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results/BRENTCMDUSD_H1_trades.csv",
        "IB H4": "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results/BRENTCMDUSD_H4_trades.csv"
    }
    analyzer = analyze_correlation_from_files(filepaths, initial_capital=10000.0)