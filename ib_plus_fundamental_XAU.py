import pandas as pd
import os
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
import yfinance as yf

# --- CONFIGURATION ---
# 1. API KEYS & PATHS
FRED_API_KEY = ''  # <-- INSERT KEY HERE
FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data"

# 2. STRATEGY SETTINGS
SYMBOLS = ["BRENTCMDUSD"] #SO 2010-2018 GOLD not bad, 2020+ good for oil
TIMEFRAMES = ["4h"]
RISK_REWARD = 2.0
INITIAL_CAPITAL = 10000.0
RISK_PERCENT = 1.0

# 3. DATE RANGE
start_date = pd.to_datetime("2019-09-20")
end_date = pd.to_datetime("2022-09-20")

# --- TIME SETTINGS ---
StartHour = 7
StartMinute = 0
EndHour = 17
EndMinute = 30


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


        # Rule 3: Tips (The big killer for Gold)
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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    try:
        start_date = df_trades['DateTime'].min()
        end_date = df_trades['DateTime'].max()

        # Download SP500
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)['Close']

        # 1. Create a Daily Equity Curve for your Strategy
        # (We need to fill the gaps between trades to compare with daily SP500)
        daily_idx = pd.date_range(start=start_date, end=end_date, freq='D')

        # Group by day and take the last balance of that day
        strategy_daily = df_trades.set_index('DateTime')['Balance'].resample('D').last()

        # Reindex to full daily range and forward fill (balance stays same on days no trades)
        strategy_equity = strategy_daily.reindex(daily_idx).ffill().dropna()

        # Align SP500 to same dates
        sp500 = sp500.reindex(strategy_equity.index).ffill().dropna()

        # 2. Normalize to Percentage Return (Start at 0%)
        # Formula: (Current_Value / Initial_Value) - 1
        strategy_pct = (strategy_equity / strategy_equity.iloc[0] - 1) * 100
        sp500_pct = (sp500 / sp500.iloc[0] - 1) * 100

        # Calculate Correlation
        correlation = strategy_pct.corr(sp500_pct)

        has_benchmark = True
    except Exception as e:
        print(f"Could not fetch S&P 500 data: {e}")
        has_benchmark = False

    # --- PLOTTING ---
    # Increased height to (20, 20) and rows to 4
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 3)

    # 1. EQUITY CURVE (Row 0)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_trades['DateTime'], df_trades['Balance'], color='green', linewidth=2)
    ax1.set_title(f"Equity Curve (Final Balance: ${df_trades['Balance'].iloc[-1]:,.2f})", fontsize=14)
    ax1.set_ylabel("Balance ($)")

    # 2. STRATEGY vs SP500 (Row 1 - NEW)
    ax_bench = fig.add_subplot(gs[1, :])
    if has_benchmark:
        ax_bench.plot(strategy_pct.index, strategy_pct, color='green', label='My Strategy', linewidth=2)
        ax_bench.plot(sp500_pct.index, sp500_pct, color='gray', linestyle='--', label='S&P 500 (Benchmark)', alpha=0.8)

        # Dynamic coloring for fill
        ax_bench.fill_between(strategy_pct.index, strategy_pct, sp500_pct,
                              where=(strategy_pct >= sp500_pct), interpolate=True, color='green', alpha=0.1)
        ax_bench.fill_between(strategy_pct.index, strategy_pct, sp500_pct,
                              where=(strategy_pct < sp500_pct), interpolate=True, color='red', alpha=0.1)

        ax_bench.set_title(f"Relative Performance vs S&P 500 (Correlation: {correlation:.2f})", fontsize=14)
        ax_bench.set_ylabel("Return (%)")
        ax_bench.legend()
    else:
        ax_bench.text(0.5, 0.5, "Benchmark Data Unavailable", ha='center')

    # 3. PROFIT DISTRIBUTION (Row 2, Col 0)
    ax2 = fig.add_subplot(gs[2, 0])
    sns.histplot(df_trades['Profit'], kde=True, ax=ax2, color='blue')
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("Distribution of Trade Profits", fontsize=12)

    # 4. MACRO SCORE IMPACT (Row 2, Col 1)
    ax3 = fig.add_subplot(gs[2, 1])
    if 'Macro_Score' in df_trades.columns:
        sns.boxplot(x='Macro_Score', y='Profit', data=df_trades, ax=ax3, palette="coolwarm")
        ax3.set_title("Profit vs. Macro Score", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No Macro_Score Data", ha='center')

    # 5. DRAWDOWN CHART (Row 2, Col 2)
    ax4 = fig.add_subplot(gs[2, 2])
    running_max = df_trades['Balance'].cummax()
    drawdown = (df_trades['Balance'] - running_max) / running_max * 100
    ax4.fill_between(df_trades['DateTime'], drawdown, 0, color='red', alpha=0.3)
    ax4.set_title(f"Drawdown % (Max: {drawdown.min():.2f}%)", fontsize=12)

    # 6. HEATMAP (Row 3, Col 0)
    ax5 = fig.add_subplot(gs[3, 0])
    pivot = df_trades.pivot_table(index='DayOfWeek', columns='Hour', values='Profit', aggfunc='sum')
    pivot = pivot.reindex(days_order)
    sns.heatmap(pivot, cmap="RdYlGn", center=0, ax=ax5, annot=False)
    ax5.set_title("Profit Heatmap (Day vs Hour)", fontsize=12)

    # 7. CUMULATIVE WINS vs LOSSES (Row 3, Col 1)
    ax6 = fig.add_subplot(gs[3, 1])
    wins = df_trades[df_trades['Profit'] > 0]['Profit'].cumsum().reset_index(drop=True)
    losses = df_trades[df_trades['Profit'] < 0]['Profit'].cumsum().reset_index(drop=True)
    ax6.plot(wins, color='green', label='Cumulative Wins')
    ax6.plot(losses, color='red', label='Cumulative Losses')
    ax6.legend()
    ax6.set_title("Win/Loss Separation", fontsize=12)

    # 8. ROLLING WIN RATE (Row 3, Col 2)
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

    # Pre-calc Technicals
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

    current_balance = INITIAL_CAPITAL
    i = 1
    n = len(df)

    while i < n - 1:
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

            if symbol == "XAUUSD" :
                if tech_direction == "Buy" and macro_score <= 0:
                    trade_allowed = True  # Buy Gold only if USD is Weak or Neutral
                elif tech_direction == "Sell" and macro_score >= 0:
                    trade_allowed = True  # Sell Gold only if USD is Strong or Neutral
            else:
                # Default (no filter for other pairs in this example)
                trade_allowed = True

            # --- 3. EXECUTION ---
            if tech_direction != "" and trade_allowed:
                risk_dist = abs(entry_price - stop_loss)
                if risk_dist > 0:
                    # Calculate position (Simplified)
                    risk_money = current_balance * (RISK_PERCENT / 100.0)
                    units = risk_money / risk_dist

                    if tech_direction == "Buy":
                        take_profit = entry_price + (risk_dist * RISK_REWARD)
                    else:
                        take_profit = entry_price - (risk_dist * RISK_REWARD)

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
                        pnl = (exit_price - entry_price) * units if tech_direction == "Buy" else (
                                                                                                             entry_price - exit_price) * units
                        current_balance += pnl
                        trades.append({
                            "Date": df.loc[i, "DateTime"],
                            "Type": tech_direction,
                            "Macro_Score": macro_score,  # Log score to see why we took trade
                            "Profit": round(pnl, 2),
                            "Balance": round(current_balance, 2)
                        })
                        i = j  # Skip processed bars

        i += 1

    return pd.DataFrame(trades)


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
                print(f"Total Trades: {len(results)}")
                print(f"Final Balance: ${results.iloc[-1]['Balance']}")
                print(f"Win Rate: {round(results[results['Profit'] > 0].shape[0] / len(results) * 100, 2)}%")
                print(f"Wins:{results[results['Profit'] > 0]['Profit'].count()}")
                print(f"Losses:{results[results['Profit'] < 0]['Profit'].count()}")
                print(f"Max Drawdown: {(INITIAL_CAPITAL - results["Balance"].min())/100}%")
                print(f"Sharpe Ratio: {round(results['Profit'].mean() / results['Profit'].std() * np.sqrt(252), 4)}")
                # Show correlation between score and trades
                print("Trades by Macro Sentiment:")
                print(results.groupby('Macro_Score')['Profit'].count())
            else:
                print("No trades.")

            perform_advanced_eda(results)

