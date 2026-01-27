import pandas as pd
import os
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

# --- CONFIGURATION ---
# 1. API KEYS & PATHS
FRED_API_KEY = 'bb21719b26aff61c740702ea701a07ed'  # <-- INSERT KEY HERE
FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data"

# 2. STRATEGY SETTINGS
SYMBOLS = ["BRENTCMDUSD"] #SO 2010-2018 GOLD not bad, 2020+ good for oil
TIMEFRAMES = ["4h"]
RISK_REWARD = 2.75
INITIAL_CAPITAL = 10000.0
RISK_PERCENT = 1.0
ATR_PERIOD = 14
ATR_TP_MULTIPLIER = 3.0
CONTRACT_SIZE = 1000
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_POINTS = 0.02

# 3. DATE RANGE
start_date = pd.to_datetime("2020-01-20")
end_date = pd.to_datetime("2024-04-20")


# --- PART 1: MACRO DATA PROCESSING ---
class MacroProcessor:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.indicators = {
            'FEDFUNDS': 'Interest_Rate',
            'CPIAUCSL': 'CPI',
            'PPIACO': 'PPI_Commodities',
            'UNRATE': 'Unemployment_Rate',
            'PAYEMS': 'NFP_Total_Jobs',
            'ICSA': 'Jobless_Claims_Initial',
            'CCSA': 'Jobless_Claims_Continuing',
            'BOPGSTB': 'Trade_Balance',
            'EXPGS': 'Exports',
            'IMPGS': 'Imports',
            'DGS2': 'US_Treasury_2Y_Yield',
            'VIXCLS': 'VIX',
            'DFII10': 'TIPS_10Y_Real_Yield',
        }

    def fetch_and_process(self):
        """Fetches data and calculates a composite 'USD_Sentiment_Score'"""
        print("Fetching Macro Data...")
        data_frames = []
        for ticker, name in self.indicators.items():
            try:
                s = self.fred.get_series(ticker)
                s.name = name
                data_frames.append(s)
            except:
                print(f"Skipping {ticker}")

        if not data_frames:
            return pd.DataFrame()

        # 1. Align and Fill
        df_macro = pd.concat(data_frames, axis=1).sort_index()
        df_macro = df_macro.ffill().dropna()

        # 2. Calculate Sentiment Score (Simple Model)
        # Logic: If Economic Indicators improve -> USD Stronger -> Gold Weaker
        # We will create a 'USD_Score'.
        # Positive Score = Strong USD (Bearish Gold).
        # Negative Score = Weak USD (Bullish Gold).

        df_macro['USD_Score'] = 0

        # Rule 1: NFP Trend (MoM)
        df_macro['NFP_Change'] = df_macro['NFP_Total_Jobs'].diff()
        df_macro.loc[df_macro['NFP_Change'] > 0, 'USD_Score'] += 1
        df_macro.loc[df_macro['NFP_Change'] < 0, 'USD_Score'] -= 1

        # Rule 2: Real Yields (The big killer for Gold)
        # Rising Yields = Strong USD
        #df_macro['Yield_Change'] = df_macro['US_Treasury_2Y_Yield'].diff()
        #df_macro.loc[df_macro['Yield_Change'] > 0, 'USD_Score'] += 1 # Weighted heavier
        #df_macro.loc[df_macro['Yield_Change'] < 0, 'USD_Score'] -= 1


        # Rule 3: Tips
        df_macro["TIPS_10Y_Real_Yield"] = df_macro["TIPS_10Y_Real_Yield"].diff()
        df_macro.loc[df_macro["TIPS_10Y_Real_Yield"] > 0, "USD_Score"] += 1
        df_macro.loc[df_macro["TIPS_10Y_Real_Yield"] < 0, "USD_Score"] -= 1


        # Rule 4: Inflation vs Deflation
        df_macro["Imports"]= df_macro["Imports"].diff()
        df_macro["Exports"] = df_macro["Exports"].diff()
        df_macro.loc[(df_macro["Imports"] < 0) & (df_macro["Exports"] >0), "USD_Score"] += 1 #deflation
        df_macro.loc[(df_macro["Imports"] > 0) & (df_macro["Exports"] <0), "USD_Score"] -= 1 #inflation

        # Rule 5: Federal Balance
        df_macro["Trade_Balance"] = df_macro["Trade_Balance"].diff()
        df_macro.loc[df_macro["Trade_Balance"] > 0, "USD_Score"] += 1
        df_macro.loc[df_macro["Trade_Balance"] < 0, "USD_Score"] -= 1

        #rule 6: unemployment rate
        #df_macro["Unemployment_Rate"] = df_macro["Unemployment_Rate"].diff()
        #df_macro.loc[df_macro["Unemployment_Rate"] > 0, "USD_Score"] -= 1
        #df_macro.loc[df_macro["Unemployment_Rate"] < 0, "USD_Score"] += 1



        # Prepare for merge (reset index to make Date a column)
        df_macro.index.name = 'DateTime'
        return df_macro.reset_index()[['DateTime', 'USD_Score']]


# --- PART 2: DATA LOADING & MERGING ---
def load_csv(symbol, timeframe):
    filepath = os.path.join(FOLDER, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, sep="\t", header=None,
                         names=["DateTime", "Open", "High", "Low", "Close", "Vol"])
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        return df
    except:
        return pd.DataFrame()


def merge_macro_to_ohlc(df_ohlc, df_macro):
    """
    Merges Macro data into OHLC data.
    Uses 'merge_asof' to map the latest available macro data to each candle.
    """
    df_ohlc = df_ohlc.sort_values("DateTime")
    df_macro = df_macro.sort_values("DateTime")

    # merge_asof looks "backward" to find the last known macro value for the current candle time
    df_merged = pd.merge_asof(df_ohlc, df_macro, on="DateTime", direction="backward")

    # Fill any remaining NaNs (e.g. at the very start) with 0 (Neutral)
    df_merged['USD_Score'] = df_merged['USD_Score'].fillna(0)

    return df_merged



# Set style
sns.set(style="darkgrid")

#eda
def perform_advanced_eda(df_trades):
    if df_trades.empty:
        print("No trades to analyze.")
        return

    # --- PRE-PROCESSING ---
    df_trades['DateTime'] = pd.to_datetime(df_trades['Date'])
    df_trades['Hour'] = df_trades['DateTime'].dt.hour
    df_trades['DayOfWeek'] = df_trades['DateTime'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # --- FETCH & PREPARE SP500 BENCHMARK ---
    print("Fetching S&P 500 data for comparison...")
    has_benchmark = False

    try:
        start_date = df_trades['DateTime'].min()
        end_date = df_trades['DateTime'].max()

        # FIX 1: Explicitly set auto_adjust=False to silence warning
        # FIX 2: Handle Timezones (tz_localize(None)) to fix "dimensions mismatch"
        sp500_df = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=False)

        if not sp500_df.empty:
            # Handle MultiIndex columns if present (common in new yfinance)
            if isinstance(sp500_df.columns, pd.MultiIndex):
                sp500 = sp500_df.xs('Close', axis=1, level=0)
                if isinstance(sp500, pd.DataFrame):
                    sp500 = sp500.iloc[:, 0]  # Ensure it's a Series
            else:
                sp500 = sp500_df['Close']

            # CRITICAL FIX: Remove Timezone info to match your Trade Data
            sp500.index = sp500.index.tz_localize(None)

            # 1. Create Strategy Equity Curve
            daily_idx = pd.date_range(start=start_date, end=end_date, freq='D')

            # Group trades by day
            strategy_daily = df_trades.set_index('DateTime')['Balance'].resample('D').last()

            # Reindex strategy to fill missing days
            strategy_equity = strategy_daily.reindex(daily_idx).ffill().dropna()

            # Align SP500 to the exact same days as strategy
            sp500 = sp500.reindex(strategy_equity.index).ffill().dropna()

            # Ensure indices match perfectly before calculation
            common_idx = strategy_equity.index.intersection(sp500.index)
            strategy_equity = strategy_equity.loc[common_idx]
            sp500 = sp500.loc[common_idx]

            if not sp500.empty and not strategy_equity.empty:
                # 2. Normalize to Percentage Return
                strategy_pct = (strategy_equity / strategy_equity.iloc[0] - 1) * 100
                sp500_pct = (sp500 / sp500.iloc[0] - 1) * 100

                correlation = strategy_pct.corr(sp500_pct)
                has_benchmark = True
            else:
                print("Data alignment resulted in empty sets.")

    except Exception as e:
        print(f"Could not fetch S&P 500 data: {e}")
        has_benchmark = False

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 3)

    # 1. EQUITY CURVE
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_trades['DateTime'], df_trades['Balance'], color='green', linewidth=2)
    ax1.set_title(f"Equity Curve (Final Balance: ${df_trades['Balance'].iloc[-1]:,.2f})", fontsize=14)
    ax1.set_ylabel("Balance ($)")

    # 2. STRATEGY vs SP500
    ax_bench = fig.add_subplot(gs[1, :])
    if has_benchmark:
        ax_bench.plot(strategy_pct.index, strategy_pct, color='green', label='My Strategy', linewidth=2)
        ax_bench.plot(sp500_pct.index, sp500_pct, color='gray', linestyle='--', label='S&P 500', alpha=0.8)

        ax_bench.fill_between(strategy_pct.index, strategy_pct, sp500_pct,
                              where=(strategy_pct >= sp500_pct), interpolate=True, color='green', alpha=0.1)
        ax_bench.fill_between(strategy_pct.index, strategy_pct, sp500_pct,
                              where=(strategy_pct < sp500_pct), interpolate=True, color='red', alpha=0.1)

        ax_bench.set_title(f"Relative Performance vs S&P 500 (Correlation: {correlation:.2f})", fontsize=14)
        ax_bench.set_ylabel("Return (%)")
        ax_bench.legend()
    else:
        ax_bench.text(0.5, 0.5, "Benchmark Data Unavailable (Check Date Range)", ha='center')

    # 3. PROFIT DISTRIBUTION
    ax2 = fig.add_subplot(gs[2, 0])
    sns.histplot(df_trades['Profit'], kde=True, ax=ax2, color='blue')
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("Distribution of Trade Profits", fontsize=12)

    # 4. MACRO SCORE IMPACT (Fixed Seaborn Warning)
    ax3 = fig.add_subplot(gs[2, 1])
    if 'Macro_Score' in df_trades.columns:
        # FIX 3: Added hue='Macro_Score' and legend=False
        sns.boxplot(x='Macro_Score', y='Profit', data=df_trades, ax=ax3, hue='Macro_Score', palette="coolwarm",
                    legend=False)
        ax3.set_title("Profit vs. Macro Score", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No Macro_Score Data", ha='center')

    # 5. DRAWDOWN CHART
    ax4 = fig.add_subplot(gs[2, 2])
    running_max = df_trades['Balance'].cummax()
    drawdown = (df_trades['Balance'] - running_max) / running_max * 100
    ax4.fill_between(df_trades['DateTime'], drawdown, 0, color='red', alpha=0.3)
    ax4.set_title(f"Drawdown % (Max: {drawdown.min():.2f}%)", fontsize=12)

    # 6. HEATMAP
    ax5 = fig.add_subplot(gs[3, 0])
    pivot = df_trades.pivot_table(index='DayOfWeek', columns='Hour', values='Profit', aggfunc='sum')
    pivot = pivot.reindex(days_order)
    sns.heatmap(pivot, cmap="RdYlGn", center=0, ax=ax5, annot=False)
    ax5.set_title("Profit Heatmap (Day vs Hour)", fontsize=12)

    # 7. CUMULATIVE WINS vs LOSSES
    ax6 = fig.add_subplot(gs[3, 1])
    wins = df_trades[df_trades['Profit'] > 0]['Profit'].cumsum().reset_index(drop=True)
    losses = df_trades[df_trades['Profit'] < 0]['Profit'].cumsum().reset_index(drop=True)
    ax6.plot(wins, color='green', label='Cumulative Wins')
    ax6.plot(losses, color='red', label='Cumulative Losses')
    ax6.legend()
    ax6.set_title("Win/Loss Separation", fontsize=12)

    # 8. ROLLING WIN RATE
    ax7 = fig.add_subplot(gs[3, 2])
    df_trades['Win'] = np.where(df_trades['Profit'] > 0, 1, 0)
    df_trades['Rolling_WR'] = df_trades['Win'].rolling(window=20).mean()
    ax7.plot(df_trades['Rolling_WR'], color='purple')
    ax7.axhline(0.5, color='gray', linestyle='--')
    ax7.set_ylim(0, 1)
    ax7.set_title("Rolling Win Rate (20 Trades)", fontsize=12)

    plt.tight_layout()
    plt.show()




# --- PART 3: STRATEGY WITH FUNDAMENTAL FILTER ---
def run_strategy_with_macro(df, symbol):
    trades = []

    df = calculate_atr(df, period=ATR_PERIOD)

    # Pre-calc Technicals
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

    current_balance = INITIAL_CAPITAL
    i = 1
    n = len(df)

    while i < n - 1:
        if pd.isna(df.loc[i, 'ATR']):
            i += 1
            continue

        # --- 1. TECHNICAL SIGNAL ---
        is_inside = df.loc[i - 1, "Is_Inside_Bar"]

        if is_inside:
            ib_high = df.loc[i - 1, "High"]
            ib_low = df.loc[i - 1, "Low"]
            curr_high = df.loc[i, "High"]
            curr_low = df.loc[i, "Low"]

            tech_direction = ""
            if curr_high > ib_high:
                tech_direction = "Buy"
                entry_price = ib_high
                stop_loss = ib_low
            elif curr_low < ib_low:
                tech_direction = "Sell"
                entry_price = ib_low
                stop_loss = ib_high

            # --- 2. FUNDAMENTAL FILTER ---
            # Get the Macro Score for this specific candle
            macro_score = df.loc[i, "USD_Score"]

            # Logic for XAUUSD (Gold vs USD)
            # If USD Score is POSITIVE (Strong USD) -> We prefer SELLING Gold.
            # If USD Score is NEGATIVE (Weak USD) -> We prefer BUYING Gold.

            trade_allowed = False

            if symbol == "BRENTCMDUSD":
                if tech_direction == "Sell" and macro_score <= 0:
                    trade_allowed = True  # Buy Gold only if USD is Weak or Neutral
                elif tech_direction == "Buy" and macro_score >= 0:
                    trade_allowed = True  # Sell Gold only if USD is Strong or Neutral
            else:
                # Default (no filter for other pairs in this example)
                trade_allowed = True

            # --- 3. EXECUTION ---
            if tech_direction != "" and trade_allowed:
                risk_dist = abs(entry_price - stop_loss)
                current_atr = df.loc[i, 'ATR']
                atr_dist_tp = current_atr * ATR_TP_MULTIPLIER
                if risk_dist > 0:
                    # Calculate position (Simplified)
                    risk_money = current_balance * (RISK_PERCENT / 100.0)
                    units = risk_money / risk_dist


                    if tech_direction == "Buy":
                        take_profit = entry_price + (risk_dist * RISK_REWARD)
                        #take_profit = entry_price + atr_dist_tp
                    else:
                        take_profit = entry_price - (risk_dist * RISK_REWARD)
                        #take_profit = entry_price - atr_dist_tp

                    # Find Exit
                    outcome = ""
                    exit_price = 0.0
                    for j in range(i, n):
                        # ... (Exit logic same as before) ...
                        bar_low = df.loc[j, "Low"]
                        bar_high = df.loc[j, "High"]

                        if tech_direction == "Buy":
                            if bar_low <= stop_loss:
                                exit_price = stop_loss;
                                outcome = "Loss";
                                break
                            elif bar_high >= take_profit:
                                exit_price = take_profit;
                                outcome = "Win";
                                break
                        elif tech_direction == "Sell":
                            if bar_high >= stop_loss:
                                exit_price = stop_loss;
                                outcome = "Loss";
                                break
                            elif bar_low <= take_profit:
                                exit_price = take_profit;
                                outcome = "Win";
                                break

                    if outcome != "":
                        # Raw PnL (without commission)
                        raw_pnl = (exit_price - entry_price) * units if tech_direction == "Buy" else (
                                                                                                                 entry_price - exit_price) * units

                        # For FUTURES: Calculate number of contracts
                        # Each CL contract = 1,000 barrels
                        # Each Brent contract = 1,000 barrels
                        CONTRACT_SIZE = 1000  # Standard for crude oil futures
                        num_contracts = units / CONTRACT_SIZE

                        # Commission: Per contract, both sides (entry + exit)
                        total_commission = (COMMISSION_PER_CONTRACT * num_contracts) * 2

                        # Net PnL
                        net_pnl = raw_pnl - total_commission
                        current_balance += net_pnl

                        trades.append({
                            "Date": df.loc[i, "DateTime"],
                            "Type": tech_direction,
                            "Macro_Score": macro_score,
                            "Raw_PnL": round(raw_pnl, 2),
                            "Commission": round(total_commission, 2),
                            "Profit": round(net_pnl, 2),
                            "Balance": round(current_balance, 2),
                            "Units": units,
                            "Contracts": round(num_contracts, 2),  # Track actual contracts
                        })
                        i = j  # Skip processed bars

        i += 1

    return pd.DataFrame(trades)


def calculate_atr(df, period=14):
    """
    Calculates Average True Range (ATR).
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    # 1. True Range Calculation
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    # Берем максимальное значение из трех вариантов
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 2. ATR Calculation (Simple Moving Average of TR)
    # Можно использовать ewm (экспоненциальное), но rolling (простое) тоже ок
    df['ATR'] = tr.rolling(window=period).mean()

    return df


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # Step 1: Get Macro Data
    # NOTE: Requires a valid key. If you don't have one now, the code will handle empty DF.
    macro_proc = MacroProcessor(FRED_API_KEY)
    df_macro = macro_proc.fetch_and_process()

    if df_macro.empty:
        print("Warning: No Macro data fetched. Strategy will run purely Technical.")
    else:
        print(f"Macro Data Ready. Rows: {len(df_macro)}")
        print(df_macro.tail(3))

    # Step 2: Run Backtest
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\n--- Testing {symbol} {tf} ---")

            # Load OHLC
            df_ohlc = load_csv(symbol, tf)
            if df_ohlc.empty: continue

            # Filter Date
            df_ohlc = df_ohlc[(df_ohlc["DateTime"] >= start_date) & (df_ohlc["DateTime"] <= end_date)]

            # MERGE: Combine Technicals with Fundamentals
            if not df_macro.empty:
                df_combined = merge_macro_to_ohlc(df_ohlc, df_macro)
            else:
                df_combined = df_ohlc
                df_combined['USD_Score'] = 0  # Default neutral

            # Run
            results = run_strategy_with_macro(df_combined, symbol)

            if not results.empty:
                wins = results[results['Profit'] > 0]['Profit'].count()
                losses = results[results['Profit'] < 0]['Profit'].count()
                total_trades = len(results)
                avg_trade_net_profit = results['Profit'].mean()
                avg_losses_trades = results[results["Profit"] < 0]['Profit'].mean()
                avg_wins_trades = results[results["Profit"] > 0]['Profit'].mean()
                win_persentage = wins / total_trades * 100
                losses_persentage = losses / total_trades * 100
                tharp_expectancy = (avg_wins_trades * win_persentage + avg_losses_trades * losses_persentage) / (
                    -avg_losses_trades)
                total_commission = results['Commission'].sum()

                print(f"Total Trades: {len(results)}")
                print(f"Total Returns: {((results['Balance'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%")
                print(f"Final Balance: ${results.iloc[-1]['Balance']}")
                print(f"Win Rate: {round(results[results['Profit'] > 0].shape[0] / len(results) * 100, 2)}%")
                print(f"Wins:{results[results['Profit'] > 0]['Profit'].count()}")
                print(f"Losses:{results[results['Profit'] < 0]['Profit'].count()}")
                print(f"Max Drawdown: {(INITIAL_CAPITAL - results['Balance'].min())/100}%")
                print(f"Sharpe Ratio: {round(results['Profit'].mean() / results['Profit'].std() * np.sqrt(252), 4)}")
                print(f"Tharp Expectancy: {round(tharp_expectancy, 2)}")
                print(f"AVG losses/trades: ${avg_losses_trades}")
                print(f"AVG wins/trades: ${avg_wins_trades}")
                print(f"Avg Net Trade Profit: ${avg_trade_net_profit}")
                print(f"Total Commission: ${total_commission}")

                # Show correlation between score and trades
                print("Trades by Macro Sentiment:")
                print(results.groupby('Macro_Score')['Profit'].count())
                output_filename = f"{symbol}_{tf}_trades.csv"
                results.to_csv(os.path.join("/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results", output_filename), index=False)
            else:
                print("No trades.")

            perform_advanced_eda(results)

