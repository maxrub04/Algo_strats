"""
Improved Modular Trading System
Separates concerns: Data Processing, Strategy Logic, Execution, Analysis
"""
import pandas as pd
import os
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration"""
    # API & Paths
    FRED_API_KEY = 'bb21719b26aff61c740702ea701a07ed'
    DATA_FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data/dukascopy/brent"
    OUTPUT_FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results"

    # Trading Parameters
    SYMBOLS = ["BRENTCMDUSD"]
    TIMEFRAMES = ["H4","H1"]
    INITIAL_CAPITAL = 10000.0
    RISK_PERCENT = 1.0

    # Strategy Parameters
    RISK_REWARD = 2.75
    ATR_PERIOD = 14
    ATR_TP_MULTIPLIER = 3.0

    # Costs
    CONTRACT_SIZE = 1000
    COMMISSION_PER_CONTRACT = 2.50
    SLIPPAGE_POINTS = 0.02

    # Date Range
    START_DATE = pd.to_datetime("2015-01-20")
    END_DATE = pd.to_datetime("2018-04-20")


# =============================================================================
# DATA PROCESSORS
# =============================================================================

class DataLoader:
    """Handles all data loading operations"""

    @staticmethod
    def load_ohlc(symbol: str, timeframe: str, folder: str) -> pd.DataFrame:
        """Load OHLC data from CSV"""
        filepath = os.path.join(folder, f"{symbol}_{timeframe}.csv")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return pd.DataFrame()

        try:
            df = pd.read_csv(filepath, sep="\t", header=0)
            df = df.rename(columns={"Time": "DateTime"})
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()

    @staticmethod
    def filter_by_date(df: pd.DataFrame, start_date: pd.Timestamp,
                       end_date: pd.Timestamp) -> pd.DataFrame:
        """Filter dataframe by date range"""
        return df[(df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)].copy()


class MacroProcessor:
    """Processes macroeconomic data"""

    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key)
        self.indicators = {
            'FEDFUNDS': 'Interest_Rate',
            'CPIAUCSL': 'CPI',
            'PAYEMS': 'NFP_Total_Jobs',
            'DFII10': 'TIPS_10Y_Real_Yield',
            'EXPGS': 'Exports',
            'IMPGS': 'Imports',
            'BOPGSTB': 'Trade_Balance',
        }

    def fetch_and_process(self) -> pd.DataFrame:
        """Fetch macro data and calculate sentiment score"""
        print("Fetching Macro Data...")
        data_frames = []

        for ticker, name in self.indicators.items():
            try:
                series = self.fred.get_series(ticker)
                series.name = name
                data_frames.append(series)
            except Exception as e:
                print(f"Skipping {ticker}: {e}")

        if not data_frames:
            return pd.DataFrame()

        # Align and fill
        df = pd.concat(data_frames, axis=1).sort_index().ffill().dropna()

        # Calculate USD Score
        df['USD_Score'] = self._calculate_usd_score(df)

        df.index.name = 'DateTime'
        return df.reset_index()[['DateTime', 'USD_Score']]

    def _calculate_usd_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate USD sentiment score from economic indicators"""
        score = pd.Series(0, index=df.index)

        # NFP Trend
        if 'NFP_Total_Jobs' in df.columns:
            nfp_change = df['NFP_Total_Jobs'].diff()
            score += np.where(nfp_change > 0, 1, 0)
            score += np.where(nfp_change < 0, -1, 0)

        # TIPS Real Yield
        if 'TIPS_10Y_Real_Yield' in df.columns:
            tips_change = df['TIPS_10Y_Real_Yield'].diff()
            score += np.where(tips_change > 0, 1, 0)
            score += np.where(tips_change < 0, -1, 0)

        # Trade Balance
        if 'Trade_Balance' in df.columns:
            trade_change = df['Trade_Balance'].diff()
            score += np.where(trade_change > 0, 1, 0)
            score += np.where(trade_change < 0, -1, 0)

        # Import/Export dynamics
        if 'Imports' in df.columns and 'Exports' in df.columns:
            imp_change = df['Imports'].diff()
            exp_change = df['Exports'].diff()
            # Deflation scenario (good for USD)
            score += np.where((imp_change < 0) & (exp_change > 0), 1, 0)
            # Inflation scenario (bad for USD)
            score += np.where((imp_change > 0) & (exp_change < 0), -1, 0)

        return score

    @staticmethod
    def merge_with_ohlc(df_ohlc: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
        """Merge macro data with OHLC data"""
        df_ohlc = df_ohlc.sort_values("DateTime")
        df_macro = df_macro.sort_values("DateTime")

        df_merged = pd.merge_asof(df_ohlc, df_macro, on="DateTime", direction="backward")
        df_merged['USD_Score'] = df_merged['USD_Score'].fillna(0)

        return df_merged


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicators:
    """Calculate technical indicators"""

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        df = df.copy()
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()

        return df

    @staticmethod
    def identify_inside_bars(df: pd.DataFrame) -> pd.DataFrame:
        """Identify inside bar patterns"""
        df = df.copy()
        df["Prev_High"] = df["High"].shift(1)
        df["Prev_Low"] = df["Low"].shift(1)
        df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])
        return df


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class InsideBarStrategy:
    """Inside Bar breakout strategy with macro filter"""

    def __init__(self, config: Config):
        self.config = config
        self.current_balance = config.INITIAL_CAPITAL

    def run(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Execute strategy on provided data

        Args:
            df: DataFrame with OHLC, ATR, Inside Bar flags, and USD_Score
            symbol: Trading symbol

        Returns:
            DataFrame with trade results
        """
        trades = []
        i = 1
        n = len(df)

        while i < n - 1:
            # Skip if ATR not available
            if pd.isna(df.loc[i, 'ATR']):
                i += 1
                continue

            # Check for inside bar breakout
            if df.loc[i - 1, "Is_Inside_Bar"]:
                trade = self._check_breakout(df, i, symbol)
                if trade:
                    trades.append(trade)
                    i = trade['exit_index']  # Skip processed bars

            i += 1

        return pd.DataFrame(trades)

    def _check_breakout(self, df: pd.DataFrame, i: int, symbol: str) -> Optional[Dict]:
        """Check for inside bar breakout and execute if conditions met"""
        ib_high = df.loc[i - 1, "High"]
        ib_low = df.loc[i - 1, "Low"]
        curr_high = df.loc[i, "High"]
        curr_low = df.loc[i, "Low"]
        macro_score = df.loc[i, "USD_Score"]

        # Determine direction
        direction = None
        entry_price = None
        stop_loss = None

        if curr_high > ib_high:
            direction = "Buy"
            entry_price = ib_high
            stop_loss = ib_low
        elif curr_low < ib_low:
            direction = "Sell"
            entry_price = ib_low
            stop_loss = ib_high
        else:
            return None

        # Apply macro filter
        if not self._check_macro_filter(symbol, direction, macro_score):
            return None

        # Calculate position size
        risk_dist = abs(entry_price - stop_loss)
        if risk_dist == 0:
            return None

        risk_money = self.current_balance * (self.config.RISK_PERCENT / 100.0)
        units = risk_money / risk_dist

        # Calculate take profit
        take_profit = self._calculate_take_profit(entry_price, stop_loss, direction)

        # Find exit
        exit_result = self._find_exit(df, i, direction, entry_price, stop_loss, take_profit)

        if exit_result is None:
            return None

        # Calculate P&L with costs
        raw_pnl = self._calculate_pnl(entry_price, exit_result['exit_price'], units, direction)
        commission = self._calculate_commission(units)
        net_pnl = raw_pnl - commission

        self.current_balance += net_pnl

        return {
            'Date': df.loc[i, 'DateTime'],
            'Type': direction,
            'Entry_Price': entry_price,
            'Exit_Price': exit_result['exit_price'],
            'Stop_Loss': stop_loss,
            'Take_Profit': take_profit,
            'Macro_Score': macro_score,
            'Raw_PnL': round(raw_pnl, 2),
            'Commission': round(commission, 2),
            'Profit': round(net_pnl, 2),
            'Balance': round(self.current_balance, 2),
            'Units': units,
            'Contracts': round(units / self.config.CONTRACT_SIZE, 2),
            'Outcome': exit_result['outcome'],
            'exit_index': exit_result['exit_index']
        }

    def _check_macro_filter(self, symbol: str, direction: str, macro_score: float) -> bool:
        """Apply macro economic filter"""
        if symbol == "BRENTCMDUSD":
            # For oil: Strong USD (positive score) -> prefer Sell
            # Weak USD (negative score) -> prefer Buy
            if direction == "Buy" and macro_score >= 0:
                return True
            elif direction == "Sell" and macro_score <= 0:
                return True
            return False

        # Default: allow all trades for other symbols
        return True

    def _calculate_take_profit(self, entry: float, stop_loss: float, direction: str) -> float:
        """Calculate take profit level"""
        risk_dist = abs(entry - stop_loss)

        if direction == "Buy":
            return entry + (risk_dist * self.config.RISK_REWARD)
        else:
            return entry - (risk_dist * self.config.RISK_REWARD)

    def _find_exit(self, df: pd.DataFrame, start_idx: int, direction: str,
                   entry: float, stop_loss: float, take_profit: float) -> Optional[Dict]:
        """Find exit point for trade"""
        n = len(df)

        for j in range(start_idx, n):
            bar_high = df.loc[j, "High"]
            bar_low = df.loc[j, "Low"]

            if direction == "Buy":
                if bar_low <= stop_loss:
                    return {'exit_price': stop_loss, 'outcome': 'Loss', 'exit_index': j}
                elif bar_high >= take_profit:
                    return {'exit_price': take_profit, 'outcome': 'Win', 'exit_index': j}

            elif direction == "Sell":
                if bar_high >= stop_loss:
                    return {'exit_price': stop_loss, 'outcome': 'Loss', 'exit_index': j}
                elif bar_low <= take_profit:
                    return {'exit_price': take_profit, 'outcome': 'Win', 'exit_index': j}

        return None

    def _calculate_pnl(self, entry: float, exit: float, units: float, direction: str) -> float:
        """Calculate raw P&L"""
        if direction == "Buy":
            return (exit - entry) * units
        else:
            return (entry - exit) * units

    def _calculate_commission(self, units: float) -> float:
        """Calculate commission costs"""
        num_contracts = units / self.config.CONTRACT_SIZE
        return (self.config.COMMISSION_PER_CONTRACT * num_contracts) * 2  # Entry + Exit


# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================

class PerformanceAnalyzer:
    """Analyze strategy performance"""

    @staticmethod
    def print_summary(results: pd.DataFrame, config: Config):
        """Print performance summary"""
        if results.empty:
            print("No trades to analyze")
            return

        wins = results[results['Profit'] > 0]
        losses = results[results['Profit'] < 0]

        total_trades = len(results)
        win_rate = len(wins) / total_trades * 100

        avg_win = wins['Profit'].mean() if len(wins) > 0 else 0
        avg_loss = losses['Profit'].mean() if len(losses) > 0 else 0

        total_wins = wins['Profit'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['Profit'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Tharp Expectancy
        if len(wins) > 0 and len(losses) > 0:
            tharp = (avg_win * win_rate - abs(avg_loss) * (100 - win_rate)) / abs(avg_loss)
        else:
            tharp = 0

        # Sharpe Ratio
        sharpe = (results['Profit'].mean() / results['Profit'].std() * np.sqrt(252)) \
            if results['Profit'].std() > 0 else 0

        # Max Drawdown
        running_max = results['Balance'].cummax()
        drawdown_pct = (results['Balance'] - running_max) / running_max * 100
        max_dd = abs(drawdown_pct.min())

        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Total Trades:            {total_trades}")
        print(f"Wins:                    {len(wins)} ({win_rate:.2f}%)")
        print(f"Losses:                  {len(losses)}")
        print(f"Profit Factor:           {profit_factor:.2f}")
        print(f"Average Win:             ${avg_win:.2f}")
        print(f"Average Loss:            ${avg_loss:.2f}")
        print(f"Tharp Expectancy:        {tharp:.2f}")
        print(f"Sharpe Ratio:            {sharpe:.3f}")
        print(f"Max Drawdown:            {max_dd:.2f}%")
        print(
            f"Total Return:            {((results['Balance'].iloc[-1] - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL * 100):.2f}%")
        print(f"Final Balance:           ${results['Balance'].iloc[-1]:,.2f}")
        print(f"Total Commission:        ${results['Commission'].sum():,.2f}")
        print("=" * 80)

    @staticmethod
    def save_results(results: pd.DataFrame, symbol: str, timeframe: str, output_folder: str):
        """Save results to CSV"""
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{symbol}_{timeframe}_trades.csv")
        results.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    config = Config()

    # Initialize processors
    data_loader = DataLoader()
    macro_processor = MacroProcessor(config.FRED_API_KEY)

    # Fetch macro data
    df_macro = macro_processor.fetch_and_process()

    if df_macro.empty:
        print("Warning: No macro data available. Running with technical signals only.")
    else:
        print(f"Macro data ready: {len(df_macro)} rows\n")

    # Run backtest for each symbol/timeframe
    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            print(f"\n{'=' * 80}")
            print(f"Testing: {symbol} {timeframe}")
            print(f"{'=' * 80}")

            # Load and prepare data
            df_ohlc = data_loader.load_ohlc(symbol, timeframe, config.DATA_FOLDER)
            if df_ohlc.empty:
                continue

            # Filter by date
            df_ohlc = data_loader.filter_by_date(df_ohlc, config.START_DATE, config.END_DATE)

            # Merge with macro data
            if not df_macro.empty:
                df_combined = macro_processor.merge_with_ohlc(df_ohlc, df_macro)
            else:
                df_combined = df_ohlc
                df_combined['USD_Score'] = 0

            # Calculate technical indicators
            df_combined = TechnicalIndicators.calculate_atr(df_combined, config.ATR_PERIOD)
            df_combined = TechnicalIndicators.identify_inside_bars(df_combined)

            # Run strategy
            strategy = InsideBarStrategy(config)
            results = strategy.run(df_combined, symbol)

            # Analyze and save results
            if not results.empty:
                PerformanceAnalyzer.print_summary(results, config)
                PerformanceAnalyzer.save_results(results, symbol, timeframe, config.OUTPUT_FOLDER)
            else:
                print("No trades generated")


if __name__ == "__main__":
    main()