"""
Walk-Forward Analysis Framework
Generic framework that accepts any strategy and performs rolling optimization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Dict, List, Tuple, Optional, Any
from itertools import product
import warnings

warnings.filterwarnings('ignore')


class WalkForwardAnalyzer:
    """
    Generic Walk-Forward Analysis framework
    Works with any strategy by accepting a strategy function
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategy_func: Callable,
                 param_grid: Dict[str, List],
                 initial_capital: float = 10000.0):
        """
        Args:
            data: OHLC data with DateTime column
            strategy_func: Function that takes (data, **params) and returns trades DataFrame
            param_grid: Dictionary of parameters to optimize {param_name: [values]}
            initial_capital: Starting capital
        """
        self.data = data.copy()
        self.strategy_func = strategy_func
        self.param_grid = param_grid
        self.initial_capital = initial_capital

        # Ensure DateTime column exists and is datetime
        if 'DateTime' in self.data.columns:
            self.data['DateTime'] = pd.to_datetime(self.data['DateTime'])
        elif 'Date' in self.data.columns:
            self.data['DateTime'] = pd.to_datetime(self.data['Date'])
        else:
            raise ValueError("Data must have 'DateTime' or 'Date' column")

        self.data = self.data.sort_values('DateTime').reset_index(drop=True)

    def run_walk_forward(self,
                         train_periods: int = 4,
                         test_periods: int = 1,
                         period_type: str = 'years',
                         optimization_metric: str = 'total_profit',
                         min_trades_required: int = 10) -> Dict:
        """
        Run Walk-Forward Analysis

        Args:
            train_periods: Number of periods for training (in-sample)
            test_periods: Number of periods for testing (out-of-sample)
            period_type: 'years', 'months', or 'quarters'
            optimization_metric: Metric to optimize ('total_profit', 'sharpe', 'calmar', etc.)
            min_trades_required: Minimum trades required in optimization

        Returns:
            Dictionary with WFA results
        """
        print("\n" + "=" * 80)
        print("WALK-FORWARD ANALYSIS")
        print("=" * 80)
        print(f"Training Period: {train_periods} {period_type}")
        print(f"Testing Period: {test_periods} {period_type}")
        print(f"Optimization Metric: {optimization_metric}")
        print(f"Parameter Grid: {self.param_grid}")
        print("=" * 80 + "\n")

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        print(f"Total parameter combinations to test: {len(param_combinations)}\n")

        # Generate windows
        windows = self._generate_windows(train_periods, test_periods, period_type)

        if not windows:
            raise ValueError("Not enough data for Walk-Forward Analysis")

        print(f"Generated {len(windows)} walk-forward windows\n")

        # Run WFA
        all_test_results = []
        window_summary = []

        for i, (train_start, train_end, test_end) in enumerate(windows, 1):
            print(f"Window {i}/{len(windows)}: Train [{train_start.date()} → {train_end.date()}] " +
                  f"Test [{train_end.date()} → {test_end.date()}]")

            # Split data
            train_data = self.data[(self.data['DateTime'] >= train_start) &
                                   (self.data['DateTime'] < train_end)].reset_index(drop=True)
            test_data = self.data[(self.data['DateTime'] >= train_end) &
                                  (self.data['DateTime'] < test_end)].reset_index(drop=True)

            if train_data.empty or test_data.empty:
                print("  ⚠️  Skipping: Insufficient data")
                continue

            # Optimization Phase (In-Sample)
            best_params, best_score = self._optimize(train_data, param_combinations,
                                                     optimization_metric, min_trades_required)

            if best_params is None:
                print("  ⚠️  Skipping: No valid parameters found")
                continue

            print(f"  ✓ Best params: {best_params} (Score: {best_score:.2f})")

            # Test Phase (Out-of-Sample)
            test_trades = self.strategy_func(test_data, **best_params)

            if test_trades is None or test_trades.empty:
                print("  ⚠️  No trades in test period")
                continue

            # Calculate test metrics
            test_metrics = self._calculate_metrics(test_trades)
            test_profit = test_metrics['total_profit']

            print(f"  → Test Result: ${test_profit:,.2f} ({len(test_trades)} trades)")

            # Store results
            test_trades['Window'] = i
            test_trades['Best_Params'] = str(best_params)
            all_test_results.append(test_trades)

            window_summary.append({
                'Window': i,
                'Train_Start': train_start,
                'Train_End': train_end,
                'Test_End': test_end,
                'Best_Params': best_params,
                'Train_Score': best_score,
                'Test_Profit': test_profit,
                'Test_Trades': len(test_trades),
                **{f'Test_{k}': v for k, v in test_metrics.items()}
            })

        if not all_test_results:
            print("\n⚠️  No valid test results generated")
            return None

        # Combine all test results
        combined_results = pd.concat(all_test_results, ignore_index=True)
        combined_results = combined_results.sort_values('Date').reset_index(drop=True)

        # Calculate equity curve
        if 'Balance' not in combined_results.columns:
            combined_results['Balance'] = self.initial_capital + combined_results['Profit'].cumsum()

        # Summary statistics
        summary = self._calculate_wfa_summary(combined_results, window_summary)

        print("\n" + "=" * 80)
        print("WALK-FORWARD ANALYSIS COMPLETE")
        print("=" * 80)
        self._print_summary(summary)

        return {
            'trades': combined_results,
            'window_summary': pd.DataFrame(window_summary),
            'summary': summary,
            'initial_capital': self.initial_capital
        }

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations from grid"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = []

        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _generate_windows(self, train_periods: int, test_periods: int,
                          period_type: str) -> List[Tuple]:
        """Generate train/test windows"""
        start_date = self.data['DateTime'].min()
        end_date = self.data['DateTime'].max()

        windows = []
        current_start = start_date

        # Convert periods to DateOffset
        if period_type == 'years':
            train_offset = pd.DateOffset(years=train_periods)
            test_offset = pd.DateOffset(years=test_periods)
            step_offset = pd.DateOffset(years=test_periods)
        elif period_type == 'months':
            train_offset = pd.DateOffset(months=train_periods)
            test_offset = pd.DateOffset(months=test_periods)
            step_offset = pd.DateOffset(months=test_periods)
        elif period_type == 'quarters':
            train_offset = pd.DateOffset(months=train_periods * 3)
            test_offset = pd.DateOffset(months=test_periods * 3)
            step_offset = pd.DateOffset(months=test_periods * 3)
        else:
            raise ValueError("period_type must be 'years', 'months', or 'quarters'")

        while True:
            train_start = current_start
            train_end = train_start + train_offset
            test_end = train_end + test_offset

            if test_end > end_date:
                break

            windows.append((train_start, train_end, test_end))
            current_start += step_offset

        return windows

    def _optimize(self, train_data: pd.DataFrame, param_combinations: List[Dict],
                  metric: str, min_trades: int) -> Tuple[Optional[Dict], float]:
        """Optimize parameters on training data"""
        best_score = -np.inf
        best_params = None

        for params in param_combinations:
            try:
                trades = self.strategy_func(train_data, **params)

                if trades is None or trades.empty or len(trades) < min_trades:
                    continue

                metrics = self._calculate_metrics(trades)
                score = metrics.get(metric, -np.inf)

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                # Skip parameters that cause errors
                continue

        return best_params, best_score

    def _calculate_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from trades"""
        profits = trades['Profit'].values

        if 'Balance' not in trades.columns:
            balance = self.initial_capital + np.cumsum(profits)
        else:
            balance = trades['Balance'].values

        # Basic metrics
        total_profit = profits.sum()
        avg_profit = profits.mean()

        # Win rate
        wins = len(trades[trades['Profit'] > 0])
        losses = len(trades[trades['Profit'] < 0])
        win_rate = wins / len(trades) if len(trades) > 0 else 0

        # Profit factor
        total_wins = trades[trades['Profit'] > 0]['Profit'].sum() if wins > 0 else 0
        total_losses = abs(trades[trades['Profit'] < 0]['Profit'].sum()) if losses > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Max Drawdown
        running_max = np.maximum.accumulate(balance)
        drawdown_pct = ((balance - running_max) / running_max * 100)
        max_dd = abs(np.min(drawdown_pct))

        # Sharpe Ratio
        sharpe = (profits.mean() / profits.std()) * np.sqrt(252) if profits.std() > 0 else 0

        # Calmar Ratio
        total_return_pct = (balance[-1] - self.initial_capital) / self.initial_capital * 100
        calmar = total_return_pct / max_dd if max_dd > 0 else 0

        return {
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'calmar': calmar,
            'num_trades': len(trades)
        }

    def _calculate_wfa_summary(self, trades: pd.DataFrame,
                               window_summary: List[Dict]) -> Dict:
        """Calculate overall WFA summary statistics"""
        metrics = self._calculate_metrics(trades)

        # Window statistics
        window_df = pd.DataFrame(window_summary)
        profitable_windows = len(window_df[window_df['Test_Profit'] > 0])
        total_windows = len(window_df)

        summary = {
            **metrics,
            'total_windows': total_windows,
            'profitable_windows': profitable_windows,
            'window_win_rate': profitable_windows / total_windows if total_windows > 0 else 0,
            'initial_capital': self.initial_capital,
            'final_capital': trades['Balance'].iloc[-1] if 'Balance' in trades.columns else
            self.initial_capital + trades['Profit'].sum(),
        }

        summary['total_return_pct'] = ((summary['final_capital'] - self.initial_capital) /
                                       self.initial_capital * 100)

        return summary

    def _print_summary(self, summary: Dict):
        """Print formatted summary"""
        print(f"\nTotal Windows:          {summary['total_windows']}")
        print(f"Profitable Windows:     {summary['profitable_windows']} " +
              f"({summary['window_win_rate'] * 100:.1f}%)")
        print(f"Total Trades:           {summary['num_trades']}")
        print(f"Win Rate:               {summary['win_rate'] * 100:.1f}%")
        print(f"Profit Factor:          {summary['profit_factor']:.2f}")
        print(f"Total Return:           {summary['total_return_pct']:.2f}%")
        print(f"Max Drawdown:           {summary['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:           {summary['sharpe']:.3f}")
        print(f"Calmar Ratio:           {summary['calmar']:.3f}")
        print(f"Final Capital:          ${summary['final_capital']:,.2f}")

    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot WFA results"""
        trades = results['trades']
        window_summary = results['window_summary']

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(trades['Date'], trades['Balance'], color='blue', linewidth=2, label='Walk-Forward Equity')
        ax1.axhline(self.initial_capital, color='green', linestyle='--', linewidth=1, label='Initial Capital')
        ax1.set_title('Walk-Forward Analysis: Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Balance ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Window Performance
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['green' if p > 0 else 'red' for p in window_summary['Test_Profit']]
        ax2.bar(window_summary['Window'], window_summary['Test_Profit'], color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Profit by Window', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Profit ($)')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Cumulative Profit by Window
        ax3 = fig.add_subplot(gs[1, 1])
        window_summary['Cumulative_Profit'] = window_summary['Test_Profit'].cumsum()
        ax3.plot(window_summary['Window'], window_summary['Cumulative_Profit'],
                 color='purple', marker='o', linewidth=2)
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_title('Cumulative Profit Across Windows', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Cumulative Profit ($)')
        ax3.grid(True, alpha=0.3)

        # 4. Trade Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(trades['Profit'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Trade Profit Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Profit ($)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Drawdown
        ax5 = fig.add_subplot(gs[2, 1])
        balance = trades['Balance'].values
        running_max = np.maximum.accumulate(balance)
        drawdown = (balance - running_max) / running_max * 100
        ax5.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax5.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Trade Number')
        ax5.set_ylabel('Drawdown (%)')
        ax5.grid(True, alpha=0.3)

        plt.suptitle('Walk-Forward Analysis Results', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()


def example_strategy_function(data: pd.DataFrame, risk_reward: float = 2.0,
                              macro_threshold: int = 0) -> pd.DataFrame:
    """
    Example strategy function for demonstration
    Your strategy should follow this signature: (data, **params) -> trades_df
    """
    # This is a placeholder - implement your actual strategy logic
    # Must return DataFrame with at least 'Profit' and 'Date' columns
    trades = []
    # ... strategy logic ...
    return pd.DataFrame(trades)


if __name__ == "__main__":
    print("Walk-Forward Analysis Framework")
    print("=" * 70)
    print("\nUsage:")
    print("  1. Define your strategy function:")
    print("     def my_strategy(data, param1, param2):")
    print("         # ... strategy logic ...")
    print("         return trades_df  # Must have 'Profit' and 'Date' columns")
    print("\n  2. Create parameter grid:")
    print("     param_grid = {'param1': [1, 2, 3], 'param2': [0.5, 1.0, 1.5]}")
    print("\n  3. Run WFA:")
    print("     wfa = WalkForwardAnalyzer(data, my_strategy, param_grid)")
    print("     results = wfa.run_walk_forward()")
    print("     wfa.plot_results(results)")