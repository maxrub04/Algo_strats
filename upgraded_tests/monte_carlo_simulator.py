"""
Monte Carlo Simulation for Trading Strategies
Analyzes risk of ruin, expected returns, and drawdowns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
import os


class MonteCarloSimulator:
    """
    Monte Carlo simulation for trading strategy analysis
    """

    def __init__(self, trades_df: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Args:
            trades_df: DataFrame with 'Profit' column
            initial_capital: Starting capital
        """
        self.trades_df = trades_df
        self.initial_capital = initial_capital
        self.pnl_pool = trades_df['Profit'].values

    def run_simulation(self,
                       num_simulations: int = 2000,
                       horizon_trades: int = 100,
                       ruin_threshold: Optional[float] = None) -> Dict:
        """
        Run Monte Carlo simulation

        Args:
            num_simulations: Number of simulation runs
            horizon_trades: Number of trades to simulate
            ruin_threshold: Balance threshold for ruin (default: 60% of initial capital)

        Returns:
            Dictionary with simulation results
        """
        if ruin_threshold is None:
            ruin_threshold = self.initial_capital * 0.6

        if len(self.pnl_pool) < 10:
            raise ValueError("Need at least 10 historical trades for reliable simulation")

        print(f"Running {num_simulations} Monte Carlo simulations...")
        print(f"Historical trades: {len(self.pnl_pool)}")
        print(f"Horizon: {horizon_trades} trades")
        print(f"Ruin threshold: ${ruin_threshold:,.2f}\n")

        # Storage for results
        final_equities = []
        max_drawdowns = []
        returns_pct = []
        ret_dd_ratios = []
        is_ruined_count = 0
        all_curves = []
        min_balances = []

        # Run simulations
        for i in range(num_simulations):
            # Randomly sample trades with replacement
            sampled_pnl = np.random.choice(self.pnl_pool, size=horizon_trades, replace=True)

            # Build equity curve
            equity_curve = np.zeros(horizon_trades + 1)
            equity_curve[0] = self.initial_capital
            equity_curve[1:] = self.initial_capital + np.cumsum(sampled_pnl)

            # Check for ruin
            min_balance = np.min(equity_curve)
            min_balances.append(min_balance)

            if min_balance < ruin_threshold:
                is_ruined_count += 1

            # Calculate metrics
            final_equity = equity_curve[-1]
            total_profit = final_equity - self.initial_capital
            ret_pct = (total_profit / self.initial_capital) * 100

            # Max Drawdown
            running_max = np.maximum.accumulate(equity_curve)
            dd_curve = (equity_curve - running_max) / running_max * 100
            max_dd = abs(np.min(dd_curve))

            # Return / Drawdown Ratio
            rd_ratio = ret_pct / max_dd if max_dd > 0 else 0

            # Store results
            final_equities.append(final_equity)
            max_drawdowns.append(max_dd)
            returns_pct.append(ret_pct)
            ret_dd_ratios.append(rd_ratio)

            # Store sample curves for visualization
            if i < 100:
                all_curves.append(equity_curve)

        # Calculate statistics
        results = {
            'num_simulations': num_simulations,
            'horizon_trades': horizon_trades,
            'initial_capital': self.initial_capital,
            'ruin_threshold': ruin_threshold,
            'probability_of_ruin_pct': (is_ruined_count / num_simulations) * 100,
            'probability_of_profit_pct': np.sum(np.array(returns_pct) > 0) / num_simulations * 100,

            # Percentile statistics
            'final_equity': {
                'worst_5pct': np.percentile(final_equities, 5),
                'below_avg_25pct': np.percentile(final_equities, 25),
                'median_50pct': np.percentile(final_equities, 50),
                'above_avg_75pct': np.percentile(final_equities, 75),
                'best_5pct': np.percentile(final_equities, 95),
                'mean': np.mean(final_equities),
                'std': np.std(final_equities),
            },

            'returns_pct': {
                'worst_5pct': np.percentile(returns_pct, 5),
                'below_avg_25pct': np.percentile(returns_pct, 25),
                'median_50pct': np.percentile(returns_pct, 50),
                'above_avg_75pct': np.percentile(returns_pct, 75),
                'best_5pct': np.percentile(returns_pct, 95),
                'mean': np.mean(returns_pct),
                'std': np.std(returns_pct),
            },

            'max_drawdown_pct': {
                'worst_5pct': np.percentile(max_drawdowns, 95),  # 95th is worst
                'below_avg_25pct': np.percentile(max_drawdowns, 75),
                'median_50pct': np.percentile(max_drawdowns, 50),
                'above_avg_75pct': np.percentile(max_drawdowns, 25),
                'best_5pct': np.percentile(max_drawdowns, 5),  # 5th is best
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
            },

            'return_dd_ratio': {
                'worst_5pct': np.percentile(ret_dd_ratios, 5),
                'below_avg_25pct': np.percentile(ret_dd_ratios, 25),
                'median_50pct': np.percentile(ret_dd_ratios, 50),
                'above_avg_75pct': np.percentile(ret_dd_ratios, 75),
                'best_5pct': np.percentile(ret_dd_ratios, 95),
                'mean': np.mean(ret_dd_ratios),
                'std': np.std(ret_dd_ratios),
            },

            'min_balance': {
                'worst_5pct': np.percentile(min_balances, 5),
                'median_50pct': np.percentile(min_balances, 50),
                'best_5pct': np.percentile(min_balances, 95),
            },

            # Raw data for plotting
            '_raw_data': {
                'final_equities': final_equities,
                'returns_pct': returns_pct,
                'max_drawdowns': max_drawdowns,
                'ret_dd_ratios': ret_dd_ratios,
                'sample_curves': all_curves,
                'min_balances': min_balances,
            }
        }

        return results

    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "=" * 70)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 70)
        print(f"Simulations Run:       {results['num_simulations']:,}")
        print(f"Trade Horizon:         {results['horizon_trades']:,} trades")
        print(f"Initial Capital:       ${results['initial_capital']:,.2f}")
        print(f"Ruin Threshold:        ${results['ruin_threshold']:,.2f}")
        print("-" * 70)
        print(f"Probability of Ruin:   {results['probability_of_ruin_pct']:.2f}%")
        print(f"Probability of Profit: {results['probability_of_profit_pct']:.2f}%")
        print("=" * 70)

        # Create summary table
        metrics_data = []
        for metric_name in ['final_equity', 'returns_pct', 'max_drawdown_pct', 'return_dd_ratio']:
            metric = results[metric_name]
            metrics_data.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Worst (5%)': f"{metric['worst_5pct']:.2f}",
                'Below Avg (25%)': f"{metric['below_avg_25pct']:.2f}",
                'Median (50%)': f"{metric['median_50pct']:.2f}",
                'Above Avg (75%)': f"{metric['above_avg_75pct']:.2f}",
                'Best (5%)': f"{metric['best_5pct']:.2f}",
                'Mean': f"{metric['mean']:.2f}",
            })

        df = pd.DataFrame(metrics_data)
        print("\n" + df.to_string(index=False))
        print("\n" + "=" * 70)

    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of Monte Carlo results
        """
        raw = results['_raw_data']

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Sample Equity Curves
        ax1 = fig.add_subplot(gs[0, :])
        for curve in raw['sample_curves'][:50]:  # Plot 50 sample curves
            ax1.plot(curve, alpha=0.3, color='blue', linewidth=0.5)
        ax1.axhline(self.initial_capital, color='green', linestyle='--', label='Initial Capital', linewidth=2)
        ax1.axhline(results['ruin_threshold'], color='red', linestyle='--', label='Ruin Threshold', linewidth=2)
        ax1.set_title('Sample Equity Curves (50 simulations)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trades')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Final Equity Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(raw['final_equities'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax2.axvline(self.initial_capital, color='green', linestyle='--', linewidth=2, label='Initial Capital')
        ax2.axvline(results['final_equity']['median_50pct'], color='orange', linestyle='--', linewidth=2,
                    label='Median')
        ax2.set_title('Final Equity Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Final Equity ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(raw['returns_pct'], bins=50, color='green', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Break Even')
        ax3.axvline(results['returns_pct']['median_50pct'], color='orange', linestyle='--', linewidth=2, label='Median')
        ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Max Drawdown Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(raw['max_drawdowns'], bins=50, color='red', alpha=0.7, edgecolor='black')
        ax4.axvline(results['max_drawdown_pct']['median_50pct'], color='orange', linestyle='--', linewidth=2,
                    label='Median')
        ax4.set_title('Max Drawdown Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Risk-Return Scatter
        ax5 = fig.add_subplot(gs[2, 0])
        scatter = ax5.scatter(raw['max_drawdowns'], raw['returns_pct'],
                              c=raw['returns_pct'], cmap='RdYlGn', alpha=0.5, s=20)
        ax5.axhline(0, color='black', linestyle='--', linewidth=1)
        ax5.set_title('Risk vs Return', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Max Drawdown (%)')
        ax5.set_ylabel('Return (%)')
        plt.colorbar(scatter, ax=ax5, label='Return (%)')
        ax5.grid(True, alpha=0.3)

        # 6. Return/DD Ratio Distribution
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(raw['ret_dd_ratios'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax6.axvline(results['return_dd_ratio']['median_50pct'], color='orange', linestyle='--', linewidth=2,
                    label='Median')
        ax6.set_title('Return/Drawdown Ratio Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Return/DD Ratio')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Percentile Analysis
        ax7 = fig.add_subplot(gs[2, 2])
        percentiles = [5, 25, 50, 75, 95]
        equity_values = [np.percentile(raw['final_equities'], p) for p in percentiles]
        colors = ['darkred', 'red', 'yellow', 'lightgreen', 'darkgreen']
        bars = ax7.barh(percentiles, equity_values, color=colors, edgecolor='black')
        ax7.axvline(self.initial_capital, color='blue', linestyle='--', linewidth=2, label='Initial Capital')
        ax7.set_title('Final Equity by Percentile', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Final Equity ($)')
        ax7.set_ylabel('Percentile')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'Monte Carlo Simulation Results - {results["num_simulations"]:,} Simulations',
                     fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def analyze_risk_of_ruin(self, thresholds: List[float] = None) -> pd.DataFrame:
        """
        Analyze risk of ruin at different threshold levels

        Args:
            thresholds: List of balance thresholds to test (as percentages of initial capital)

        Returns:
            DataFrame with ruin probabilities at each threshold
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        print("Analyzing Risk of Ruin at different thresholds...")

        ruin_data = []

        for threshold_pct in thresholds:
            threshold = self.initial_capital * threshold_pct
            results = self.run_simulation(num_simulations=1000, ruin_threshold=threshold)

            ruin_data.append({
                'Threshold (% of Capital)': f"{threshold_pct * 100:.0f}%",
                'Threshold ($)': f"${threshold:,.2f}",
                'Risk of Ruin (%)': f"{results['probability_of_ruin_pct']:.2f}%",
            })

        df = pd.DataFrame(ruin_data)
        print("\n" + df.to_string(index=False))

        return df


def run_monte_carlo_from_file(filepath: str,
                              initial_capital: float = 10000.0,
                              num_simulations: int = 2000,
                              horizon_trades: int = 100,
                              ruin_threshold: Optional[float] = None,
                              plot: bool = True,
                              save_plot: Optional[str] = None) -> Dict:
    """
    Run Monte Carlo simulation from trades file

    Args:
        filepath: Path to CSV or JSON file with trades
        initial_capital: Starting capital
        num_simulations: Number of simulations to run
        horizon_trades: Number of trades to simulate
        ruin_threshold: Balance threshold for ruin
        plot: Whether to create visualizations
        save_plot: Path to save plot (optional)

    Returns:
        Dictionary with simulation results
    """
    # Load trades
    if filepath.endswith('.csv'):
        trades_df = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        trades_df = pd.read_json(filepath)
    else:
        raise ValueError("File must be .csv or .json")

    if 'Profit' not in trades_df.columns:
        raise ValueError("Trades file must have 'Profit' column")

    # Create simulator and run
    simulator = MonteCarloSimulator(trades_df, initial_capital)
    results = simulator.run_simulation(num_simulations, horizon_trades, ruin_threshold)

    # Print results
    simulator.print_results(results)

    # Plot if requested
    if plot:
        simulator.plot_results(results, save_path=save_plot)

    return results


if __name__ == "__main__":
    """print("Monte Carlo Simulator for Trading Strategies")
    print("=" * 70)
    print("\nUsage:")
    print("  from monte_carlo_simulator import run_monte_carlo_from_file")
    print("  results = run_monte_carlo_from_file('trades.csv', initial_capital=10000)")
    print("\nOr create custom simulation:")
    print("  simulator = MonteCarloSimulator(trades_df, initial_capital=10000)")
    print("  results = simulator.run_simulation(num_simulations=2000)")
    print("  simulator.print_results(results)")
    print("  simulator.plot_results(results)")"""
    file="/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results/BRENTCMDUSD_4h_trades.csv"
    results = run_monte_carlo_from_file(file, initial_capital=10000)