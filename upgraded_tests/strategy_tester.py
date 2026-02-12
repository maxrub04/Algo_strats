"""
Unified Strategy Testing Framework
One-stop testing suite for trading strategies
"""
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List
import json

# Import our testing modules
try:
    from metrics_calculator import StrategyMetrics
    from monte_carlo_simulator import MonteCarloSimulator
    from correlation_analyzer import StrategyCorrelationAnalyzer
    from walk_forward_analyzer import WalkForwardAnalyzer
except ImportError:
    print("Warning: Some testing modules not found. Make sure all modules are in the same directory.")


class StrategyTester:
    """
    Comprehensive testing framework for trading strategies
    Performs all tests from a single entry point
    """

    def __init__(self, trades_file: str, initial_capital: float = 10000.0):
        """
        Args:
            trades_file: Path to CSV/JSON file with trade history
            initial_capital: Starting capital
        """
        self.trades_file = trades_file
        self.initial_capital = initial_capital

        # Load trades
        if trades_file.endswith('.csv'):
            self.trades = pd.read_csv(trades_file)
        elif trades_file.endswith('.json'):
            self.trades = pd.read_json(trades_file)
        else:
            raise ValueError("File must be .csv or .json")

        # Validate required columns
        if 'Profit' not in self.trades.columns:
            raise ValueError("Trades file must have 'Profit' column")

        print(f"Loaded {len(self.trades)} trades from {os.path.basename(trades_file)}")

    def run_all_tests(self, output_dir: Optional[str] = None) -> Dict:
        """
        Run complete testing suite

        Args:
            output_dir: Directory to save results (optional)

        Returns:
            Dictionary with all test results
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE STRATEGY TESTING SUITE")
        print("=" * 80)

        results = {}

        # 1. Basic Metrics
        print("\n[1/3] Calculating Performance Metrics...")
        metrics = self._run_metrics_analysis()
        results['metrics'] = metrics

        # 2. Monte Carlo Simulation
        print("\n[2/3] Running Monte Carlo Simulation...")
        mc_results = self._run_monte_carlo()
        results['monte_carlo'] = mc_results

        # 3. Independence Tests (Durbin-Watson, etc.)
        print("\n[3/3] Running Independence Tests...")
        independence = self._run_independence_tests()
        results['independence'] = independence

        # Save results if output directory specified
        if output_dir:
            self._save_results(results, output_dir)

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETE")
        print("=" * 80)

        return results

    def _run_metrics_analysis(self) -> Dict:
        """Run comprehensive metrics analysis"""
        metrics_calc = StrategyMetrics(self.trades, self.initial_capital)
        metrics_calc.print_report()

        all_metrics = metrics_calc.calculate_all_metrics()
        all_metrics.update(metrics_calc.calculate_slippage_commission())
        all_metrics.update(metrics_calc.calculate_depth_fuzziness())

        return all_metrics

    def _run_monte_carlo(self, num_simulations: int = 2000,
                         horizon_trades: int = 100) -> Dict:
        """Run Monte Carlo simulation"""
        simulator = MonteCarloSimulator(self.trades, self.initial_capital)

        # Run simulation
        mc_results = simulator.run_simulation(
            num_simulations=num_simulations,
            horizon_trades=min(horizon_trades, len(self.trades)),
            ruin_threshold=self.initial_capital * 0.6
        )

        # Print results
        simulator.print_results(mc_results)

        # Plot results
        simulator.plot_results(mc_results)

        return {
            'probability_of_ruin': mc_results['probability_of_ruin_pct'],
            'probability_of_profit': mc_results['probability_of_profit_pct'],
            'median_return': mc_results['returns_pct']['median_50pct'],
            'median_max_drawdown': mc_results['max_drawdown_pct']['median_50pct'],
            'worst_case_return': mc_results['returns_pct']['worst_5pct'],
            'best_case_return': mc_results['returns_pct']['best_5pct'],
        }

    def _run_independence_tests(self) -> Dict:
        """Run statistical independence tests"""
        from statsmodels.stats.stattools import durbin_watson
        from scipy import stats

        profits = self.trades['Profit'].values
        nonzero_profits = profits[profits != 0]

        results = {}

        # 1. Durbin-Watson Test
        if len(nonzero_profits) > 2:
            dw_stat = durbin_watson(nonzero_profits)
            results['durbin_watson'] = {
                'statistic': dw_stat,
                'is_independent': 1.5 < dw_stat < 2.5,
                'interpretation': 'Independent' if 1.5 < dw_stat < 2.5 else 'Correlated'
            }

        # 2. Ljung-Box Test
        if len(nonzero_profits) > 10:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(nonzero_profits, lags=10, return_df=True)
            results['ljung_box'] = {
                'statistic': lb_result['lb_stat'].iloc[-1],
                'p_value': lb_result['lb_pvalue'].iloc[-1],
                'is_independent': lb_result['lb_pvalue'].iloc[-1] > 0.05
            }

        # 3. Runs Test
        binary = (nonzero_profits > 0).astype(int)
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i - 1]:
                runs += 1

        n_wins = np.sum(binary)
        n_losses = len(binary) - n_wins

        if n_wins > 0 and n_losses > 0:
            expected_runs = (2 * n_wins * n_losses) / len(binary) + 1
            variance = (2 * n_wins * n_losses * (2 * n_wins * n_losses - len(binary))) / \
                       (len(binary) ** 2 * (len(binary) - 1))

            z_score = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            results['runs_test'] = {
                'z_score': z_score,
                'p_value': p_value,
                'is_random': p_value > 0.05
            }

        # 4. Autocorrelation
        if len(nonzero_profits) > 1:
            autocorr = np.corrcoef(nonzero_profits[:-1], nonzero_profits[1:])[0, 1]
            results['autocorrelation'] = {
                'lag1': autocorr if not np.isnan(autocorr) else 0,
                'is_independent': abs(autocorr) < 0.3 if not np.isnan(autocorr) else True
            }

        # Print summary
        print("\nINDEPENDENCE TEST RESULTS")
        print("-" * 80)

        if 'durbin_watson' in results:
            dw = results['durbin_watson']
            print(f"Durbin-Watson:  {dw['statistic']:.4f} [{dw['interpretation']}]")

        if 'ljung_box' in results:
            lb = results['ljung_box']
            status = "✓ Independent" if lb['is_independent'] else "✗ NOT Independent"
            print(f"Ljung-Box:      p={lb['p_value']:.4f} [{status}]")

        if 'runs_test' in results:
            rt = results['runs_test']
            status = "✓ Random" if rt['is_random'] else "✗ NOT Random"
            print(f"Runs Test:      p={rt['p_value']:.4f} [{status}]")

        if 'autocorrelation' in results:
            ac = results['autocorrelation']
            status = "✓ Independent" if ac['is_independent'] else "✗ Autocorrelated"
            print(f"Autocorrelation: {ac['lag1']:.4f} [{status}]")

        return results

    def _save_results(self, results: Dict, output_dir: str):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']]).T
        metrics_df.columns = ['Value']
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'))

        # Save Monte Carlo summary
        mc_df = pd.DataFrame([results['monte_carlo']]).T
        mc_df.columns = ['Value']
        mc_df.to_csv(os.path.join(output_dir, 'monte_carlo_summary.csv'))

        # Save independence tests
        with open(os.path.join(output_dir, 'independence_tests.json'), 'w') as f:
            json.dump(results['independence'], f, indent=2)

        print(f"\n✓ Results saved to: {output_dir}")

    def generate_report(self, output_file: str = 'strategy_report.txt'):
        """Generate a comprehensive text report"""
        results = self.run_all_tests()

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE STRATEGY TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            for key, value in results['metrics'].items():
                if isinstance(value, float):
                    if 'pct' in key or 'rate' in key:
                        f.write(f"{key:.<50} {value:>10.2f}%\n")
                    elif abs(value) < 100:
                        f.write(f"{key:.<50} {value:>10.4f}\n")
                    else:
                        f.write(f"{key:.<50} ${value:>10,.2f}\n")
                else:
                    f.write(f"{key:.<50} {value:>10}\n")

            # Monte Carlo
            f.write("\n\nMONTE CARLO SIMULATION\n")
            f.write("-" * 80 + "\n")
            for key, value in results['monte_carlo'].items():
                f.write(f"{key:.<50} {value:>10.2f}\n")

            # Independence
            f.write("\n\nINDEPENDENCE TESTS\n")
            f.write("-" * 80 + "\n")
            for test, values in results['independence'].items():
                f.write(f"\n{test}:\n")
                for key, value in values.items():
                    f.write(f"  {key}: {value}\n")

        print(f"\n✓ Report saved to: {output_file}")


def quick_test(trades_file: str, initial_capital: float = 10000.0):
    """
    Quick testing function for immediate results

    Args:
        trades_file: Path to trades CSV/JSON
        initial_capital: Starting capital
    """
    tester = StrategyTester(trades_file, initial_capital)
    results = tester.run_all_tests()
    return results


if __name__ == "__main__":
   """ print("Unified Strategy Testing Framework")
    print("=" * 70)
    print("\nQuick Start:")
    print("  from strategy_tester import quick_test")
    print("  results = quick_test('my_trades.csv', initial_capital=10000)")
    print("\nAdvanced Usage:")
    print("  tester = StrategyTester('my_trades.csv', initial_capital=10000)")
    print("  results = tester.run_all_tests(output_dir='./test_results')")
    print("  tester.generate_report('report.txt')")"""
   file="/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results/BRENTCMDUSD_4h_trades.csv"
   results=quick_test(file)
