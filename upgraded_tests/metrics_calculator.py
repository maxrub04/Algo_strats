"""
Trading Strategy Metrics Calculator
Calculates comprehensive performance metrics from trade data
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional


class StrategyMetrics:
    """
    Comprehensive metrics calculator for trading strategies
    Accepts trades DataFrame with 'Profit' column (and optionally 'Balance', 'Date', 'Commission', etc.)
    """

    def __init__(self, trades_df: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Args:
            trades_df: DataFrame with at least 'Profit' column
            initial_capital: Starting capital
        """
        self.trades = trades_df.copy()
        self.initial_capital = initial_capital

        # Ensure we have a Balance column
        if 'Balance' not in self.trades.columns:
            self.trades['Balance'] = initial_capital + self.trades['Profit'].cumsum()

        # Ensure we have Date column for time-based calculations
        if 'Date' in self.trades.columns:
            self.trades['Date'] = pd.to_datetime(self.trades['Date'])

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all available metrics"""
        metrics = {}

        # Basic metrics
        metrics.update(self._basic_metrics())

        # Win/Loss metrics
        metrics.update(self._win_loss_metrics())

        # Risk metrics
        metrics.update(self._risk_metrics())

        # Advanced metrics
        metrics.update(self._advanced_metrics())

        # Time-based metrics
        if 'Date' in self.trades.columns:
            metrics.update(self._time_metrics())

        return metrics

    def _basic_metrics(self) -> Dict[str, float]:
        """Basic profitability metrics"""
        profits = self.trades['Profit'].values

        total_trades = len(self.trades)
        total_net_profit = profits.sum()
        avg_trade_net_profit = profits.mean()

        return {
            'total_trades': total_trades,
            'total_net_profit': total_net_profit,
            'avg_trade_net_profit': avg_trade_net_profit,
            'total_return_pct': (total_net_profit / self.initial_capital) * 100,
        }

    def _win_loss_metrics(self) -> Dict[str, float]:
        """Win/Loss analysis"""
        profits = self.trades['Profit'].values

        winning_trades = self.trades[self.trades['Profit'] > 0]
        losing_trades = self.trades[self.trades['Profit'] < 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        total_trades = len(self.trades)

        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
        loss_rate = (num_losses / total_trades * 100) if total_trades > 0 else 0

        avg_win = winning_trades['Profit'].mean() if num_wins > 0 else 0
        avg_loss = losing_trades['Profit'].mean() if num_losses > 0 else 0

        total_wins = winning_trades['Profit'].sum() if num_wins > 0 else 0
        total_losses = abs(losing_trades['Profit'].sum()) if num_losses > 0 else 0

        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        return {
            'num_wins': num_wins,
            'num_losses': num_losses,
            'win_rate_pct': win_rate,
            'loss_rate_pct': loss_rate,
            'avg_winning_trade': avg_win,
            'avg_losing_trade': avg_loss,
            'profit_factor': profit_factor,
            'total_wins': total_wins,
            'total_losses': total_losses,
        }

    def _risk_metrics(self) -> Dict[str, float]:
        """Risk and drawdown metrics"""
        balance = self.trades['Balance'].values
        profits = self.trades['Profit'].values

        # Max Drawdown
        running_max = np.maximum.accumulate(balance)
        drawdown = (balance - running_max) / running_max * 100
        max_drawdown = abs(np.min(drawdown))

        # Average Drawdown
        avg_drawdown = abs(np.mean(drawdown[drawdown < 0])) if any(drawdown < 0) else 0

        # Max Drawdown in dollars
        max_dd_dollars = np.min(balance - running_max)

        # Calmar Ratio (Annual Return / Max Drawdown)
        total_return_pct = (balance[-1] - self.initial_capital) / self.initial_capital * 100
        calmar_ratio = (total_return_pct / max_drawdown) if max_drawdown > 0 else 0

        # Sharpe Ratio (assuming daily trading, annualized)
        if len(profits) > 1:
            sharpe = (profits.mean() / profits.std()) * np.sqrt(252) if profits.std() > 0 else 0
        else:
            sharpe = 0

        return {
            'max_drawdown_pct': max_drawdown,
            'max_drawdown_dollars': abs(max_dd_dollars),
            'avg_drawdown_pct': avg_drawdown,
            'calmar_ratio': calmar_ratio,
            'sharpe_ratio': sharpe,
        }

    def _advanced_metrics(self) -> Dict[str, float]:
        """Advanced statistical metrics"""
        profits = self.trades['Profit'].values
        balance = self.trades['Balance'].values

        # Tharp Expectancy
        wins = self.trades[self.trades['Profit'] > 0]
        losses = self.trades[self.trades['Profit'] < 0]

        if len(wins) > 0 and len(losses) > 0:
            avg_win = wins['Profit'].mean()
            avg_loss = abs(losses['Profit'].mean())
            win_pct = len(wins) / len(self.trades)
            loss_pct = len(losses) / len(self.trades)

            tharp_expectancy = (avg_win * win_pct - avg_loss * loss_pct) / avg_loss if avg_loss > 0 else 0
        else:
            tharp_expectancy = 0

        # Kelly Criterion
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(self.trades)
            avg_win_loss_ratio = abs(wins['Profit'].mean() / losses['Profit'].mean())
            kelly = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        else:
            kelly = 0

        # Equity Curve Slope (linear regression)
        if len(balance) > 1:
            x = np.arange(len(balance))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, balance)
            equity_slope = slope
            equity_r_squared = r_value ** 2
        else:
            equity_slope = 0
            equity_r_squared = 0

        # Consecutive wins/losses
        win_streak = 0
        loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for profit in profits:
            if profit > 0:
                current_win_streak += 1
                current_loss_streak = 0
                win_streak = max(win_streak, current_win_streak)
            elif profit < 0:
                current_loss_streak += 1
                current_win_streak = 0
                loss_streak = max(loss_streak, current_loss_streak)

        return {
            'tharp_expectancy': tharp_expectancy,
            'kelly_criterion': kelly,
            'equity_curve_slope': equity_slope,
            'equity_curve_r_squared': equity_r_squared,
            'max_consecutive_wins': win_streak,
            'max_consecutive_losses': loss_streak,
        }

    def _time_metrics(self) -> Dict[str, float]:
        """Time-based metrics"""
        if 'Date' not in self.trades.columns or len(self.trades) < 2:
            return {}

        dates = self.trades['Date']
        balance = self.trades['Balance'].values

        # Trading period
        total_days = (dates.max() - dates.min()).days
        total_years = total_days / 365.25

        # Annual metrics
        total_return = (balance[-1] - self.initial_capital) / self.initial_capital
        annual_return_pct = (total_return / total_years * 100) if total_years > 0 else 0

        # Average trades per year
        trades_per_year = len(self.trades) / total_years if total_years > 0 else 0

        # Equity curve flat periods (periods with no new high)
        running_max = pd.Series(balance).expanding().max()
        flat_periods = (balance < running_max.values).sum()
        flat_period_pct = (flat_periods / len(balance) * 100) if len(balance) > 0 else 0

        return {
            'trading_period_days': total_days,
            'trading_period_years': total_years,
            'annual_return_pct': annual_return_pct,
            'trades_per_year': trades_per_year,
            'equity_flat_period_pct': flat_period_pct,
        }

    def calculate_slippage_commission(self) -> Dict[str, float]:
        """Calculate slippage and commission if available"""
        metrics = {}

        if 'Commission' in self.trades.columns:
            metrics['total_commission'] = self.trades['Commission'].sum()
            metrics['avg_commission_per_trade'] = self.trades['Commission'].mean()

        if 'Slippage' in self.trades.columns:
            metrics['total_slippage'] = self.trades['Slippage'].sum()
            metrics['avg_slippage_per_trade'] = self.trades['Slippage'].mean()

        return metrics

    def calculate_depth_fuzziness(self) -> Dict[str, float]:
        """
        Calculate Depth and Fuzziness metrics
        Depth: How well-defined is the strategy (consistency of results)
        Fuzziness: Randomness/noise in equity curve
        """
        balance = self.trades['Balance'].values
        profits = self.trades['Profit'].values

        # Depth: Coefficient of variation of profits (lower = more consistent)
        if profits.std() > 0:
            depth = abs(profits.mean() / profits.std())
        else:
            depth = 0

        # Fuzziness: Autocorrelation of profits (measure of randomness)
        if len(profits) > 1:
            # Calculate lag-1 autocorrelation
            fuzziness = np.corrcoef(profits[:-1], profits[1:])[0, 1]
            fuzziness = abs(fuzziness) if not np.isnan(fuzziness) else 0
        else:
            fuzziness = 0

        return {
            'depth': depth,
            'fuzziness': fuzziness,
        }

    def get_summary_report(self) -> pd.DataFrame:
        """Generate a formatted summary report"""
        all_metrics = self.calculate_all_metrics()
        all_metrics.update(self.calculate_slippage_commission())
        all_metrics.update(self.calculate_depth_fuzziness())

        # Format as DataFrame for nice display
        metrics_df = pd.DataFrame([all_metrics]).T
        metrics_df.columns = ['Value']

        return metrics_df

    def print_report(self):
        """Print formatted metrics report"""
        print("\n" + "=" * 70)
        print("TRADING STRATEGY PERFORMANCE REPORT")
        print("=" * 70)

        all_metrics = self.calculate_all_metrics()
        all_metrics.update(self.calculate_slippage_commission())
        all_metrics.update(self.calculate_depth_fuzziness())

        # Categorize metrics
        categories = {
            'BASIC METRICS': [
                'total_trades', 'total_net_profit', 'avg_trade_net_profit', 'total_return_pct'
            ],
            'WIN/LOSS ANALYSIS': [
                'num_wins', 'num_losses', 'win_rate_pct', 'avg_winning_trade',
                'avg_losing_trade', 'profit_factor'
            ],
            'RISK METRICS': [
                'max_drawdown_pct', 'max_drawdown_dollars', 'avg_drawdown_pct',
                'sharpe_ratio', 'calmar_ratio'
            ],
            'ADVANCED METRICS': [
                'tharp_expectancy', 'kelly_criterion', 'equity_curve_slope',
                'equity_curve_r_squared', 'max_consecutive_wins', 'max_consecutive_losses'
            ],
            'QUALITY METRICS': [
                'depth', 'fuzziness'
            ]
        }

        # Add time metrics if available
        if 'trading_period_years' in all_metrics:
            categories['TIME-BASED METRICS'] = [
                'trading_period_days', 'trading_period_years', 'annual_return_pct',
                'trades_per_year', 'equity_flat_period_pct'
            ]

        # Add cost metrics if available
        if 'total_commission' in all_metrics:
            categories['COST ANALYSIS'] = [
                'total_commission', 'avg_commission_per_trade'
            ]
        if 'total_slippage' in all_metrics:
            if 'COST ANALYSIS' not in categories:
                categories['COST ANALYSIS'] = []
            categories['COST ANALYSIS'].extend(['total_slippage', 'avg_slippage_per_trade'])

        # Print by category
        for category, metric_names in categories.items():
            print(f"\n{category}")
            print("-" * 70)
            for metric in metric_names:
                if metric in all_metrics:
                    value = all_metrics[metric]
                    # Format based on metric type
                    if 'pct' in metric or 'rate' in metric:
                        print(f"  {metric.replace('_', ' ').title():.<50} {value:>10.2f}%")
                    elif isinstance(value, float) and abs(value) < 100:
                        print(f"  {metric.replace('_', ' ').title():.<50} {value:>10.4f}")
                    elif isinstance(value, float):
                        print(f"  {metric.replace('_', ' ').title():.<50} ${value:>10,.2f}")
                    else:
                        print(f"  {metric.replace('_', ' ').title():.<50} {value:>10}")

        print("=" * 70 + "\n")


def calculate_metrics_from_file(filepath: str, initial_capital: float = 10000.0) -> StrategyMetrics:
    """
    Load trades from CSV/JSON and calculate metrics

    Args:
        filepath: Path to CSV or JSON file with trades
        initial_capital: Starting capital

    Returns:
        StrategyMetrics object
    """
    if filepath.endswith('.csv'):
        trades_df = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        trades_df = pd.read_json(filepath)
    else:
        raise ValueError("File must be .csv or .json")

    return StrategyMetrics(trades_df, initial_capital)


if __name__ == "__main__":
    # Example usage
    """print("Strategy Metrics Calculator")
    print("Import this module to calculate comprehensive trading metrics")
    print("\nExample:")
    print("  from metrics_calculator import calculate_metrics_from_file")
    print("  metrics = calculate_metrics_from_file('my_trades.csv', initial_capital=10000)")
    print("  metrics.print_report()")"""
    file = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results/BRENTCMDUSD_4h_trades.csv"
    metrics = calculate_metrics_from_file(file)
    metrics.print_report()