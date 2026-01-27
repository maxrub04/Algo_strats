import pandas as pd
import os
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
FRED_API_KEY = 'bb21719b26aff61c740702ea701a07ed'
FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data"

# Strategy Settings
SYMBOLS = ["BRENTCMDUSD"]
TIMEFRAMES = ["4h"]
ATR_PERIOD = 14

# We test how price behaves after X bars
PREDICTION_HORIZONS = [1, 4, 12, 24]  # 4h bars -> 4h, 16h, 48h, 96h

start_date = pd.to_datetime("2022-01-20")
end_date = pd.to_datetime("2025-04-20")


# --- MACRO PROCESSING (REUSED) ---
class MacroProcessor:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.indicators = {
            'FEDFUNDS': 'Interest_Rate',
            'CPIAUCSL': 'CPI',
            'T10Y2Y': 'Yield_Curve_10Y_2Y',  # Example replacement or addition
            'PAYEMS': 'NFP_Total_Jobs',
            'EXPGS': 'Exports',
            'IMPGS': 'Imports',
            'DFII10': 'TIPS_10Y_Real_Yield',
        }

    def fetch_and_process(self):
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

        df_macro = pd.concat(data_frames, axis=1).sort_index()
        df_macro = df_macro.ffill().dropna()

        # Simple Logic Reused
        df_macro['USD_Score'] = 0

        # Rule: TIPS Real Yield
        if 'TIPS_10Y_Real_Yield' in df_macro.columns:
            df_macro["TIPS_Diff"] = df_macro["TIPS_10Y_Real_Yield"].diff()
            df_macro.loc[df_macro["TIPS_Diff"] > 0, "USD_Score"] += 1
            df_macro.loc[df_macro["TIPS_Diff"] < 0, "USD_Score"] -= 1

        # Rule: Trade Balance Proxy
        if 'Exports' in df_macro.columns and 'Imports' in df_macro.columns:
            df_macro["Exp_Diff"] = df_macro["Exports"].diff()
            df_macro["Imp_Diff"] = df_macro["Imports"].diff()
            # Net Exports improving -> USD Strong
            df_macro.loc[(df_macro["Exp_Diff"] > 0) & (df_macro["Imp_Diff"] < 0), "USD_Score"] += 1
            df_macro.loc[(df_macro["Exp_Diff"] < 0) & (df_macro["Imp_Diff"] > 0), "USD_Score"] -= 1

        df_macro.index.name = 'DateTime'
        return df_macro.reset_index()[['DateTime', 'USD_Score']]


# --- DATA HELPERS ---
def load_csv(symbol, timeframe):
    filepath = os.path.join(FOLDER, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, sep="\t", header=None,
                         names=["DateTime", "Open", "High", "Low", "Close", "Vol"])
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()


def calculate_atr(df, period=14):
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


def merge_macro_to_ohlc(df_ohlc, df_macro):
    df_ohlc = df_ohlc.sort_values("DateTime")
    df_macro = df_macro.sort_values("DateTime")
    df_merged = pd.merge_asof(df_ohlc, df_macro, on="DateTime", direction="backward")
    df_merged['USD_Score'] = df_merged['USD_Score'].fillna(0)
    return df_merged


# --- ENTRY ANALYSIS ENGINE ---
def analyze_entries(df, symbol):
    """
    Analyzes the quality of entries without applying stop loss/take profit management.
    Calculates MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion) over horizons.
    """
    print(f"Analyzing Entries for {symbol}...")

    df = calculate_atr(df, period=ATR_PERIOD)
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

    entries = []
    n = len(df)

    # Iterate through bars
    for i in range(1, n - max(PREDICTION_HORIZONS) - 1):
        # 1. Check Setup (Inside Bar on previous candle)
        is_inside = df.loc[i - 1, "Is_Inside_Bar"]

        if not is_inside:
            continue

        ib_high = df.loc[i - 1, "High"]
        ib_low = df.loc[i - 1, "Low"]
        curr_high = df.loc[i, "High"]
        curr_low = df.loc[i, "Low"]
        macro_score = df.loc[i, "USD_Score"]

        # 2. Determine Direction (Breakout)
        direction = None
        entry_price = 0

        if curr_high > ib_high:
            direction = "Buy"
            entry_price = ib_high
        elif curr_low < ib_low:
            direction = "Sell"
            entry_price = ib_low

        if direction is None:
            continue

        # 3. Filter Logic (Apply same macro logic as original strategy)
        trade_allowed = True
        if symbol == "BRENTCMDUSD":
            if direction == "Sell" and macro_score > 0:  # Strong USD usually bad for commodities, so Sell is good?
                # Original logic: Sell allowed if USD Score Positive (Strong USD -> Weak Gold/Oil)
                # But user code said: if tech_direction == "Sell" and macro_score <= 0: trade_allowed = True
                # Let's stick strictly to user logic:
                if macro_score > 0: trade_allowed = False  # Block Sell if USD is Strong? (Wait, usually Strong USD = Sell Asset)
                # Re-reading user logic:
                # Sell Gold only if USD is Strong or Neutral (macro >= 0) -> trade_allowed = True
                pass

                # Let's apply a simplified filter for the test to see "Raw" vs "Filtered"
            # We will store the Macro Score and filter in post-analysis

        entry_data = {
            "Date": df.loc[i, "DateTime"],
            "Type": direction,
            "Entry_Price": entry_price,
            "Macro_Score": macro_score,
            "ATR": df.loc[i, "ATR"]
        }

        # 4. Calculate Forward Returns (MFE/MAE)
        # Look ahead for each horizon
        for h in PREDICTION_HORIZONS:
            future_slice = df.iloc[i + 1: i + 1 + h]

            if future_slice.empty:
                continue

            future_high = future_slice["High"].max()
            future_low = future_slice["Low"].min()
            close_at_h = future_slice["Close"].iloc[-1]

            if direction == "Buy":
                # MFE: How high did it go relative to entry?
                mfe_pct = (future_high - entry_price) / entry_price * 100
                # MAE: How low did it drop relative to entry? (Drawdown)
                mae_pct = (future_low - entry_price) / entry_price * 100
                # Final result at horizon
                ret_pct = (close_at_h - entry_price) / entry_price * 100

            else:  # Sell
                # MFE: How low did it drop? (Profit for short)
                mfe_pct = (entry_price - future_low) / entry_price * 100
                # MAE: How high did it go? (Drawdown for short)
                mae_pct = (entry_price - future_high) / entry_price * 100
                # Final result
                ret_pct = (entry_price - close_at_h) / entry_price * 100

            entry_data[f"MFE_{h}"] = mfe_pct
            entry_data[f"MAE_{h}"] = mae_pct
            entry_data[f"Return_{h}"] = ret_pct

        entries.append(entry_data)

    return pd.DataFrame(entries)


def plot_entry_stats(df_entries):
    if df_entries.empty:
        print("No entries found.")
        return

    # Set up layout
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2)

    # 1. Boxplot of Returns per Horizon
    ax1 = fig.add_subplot(gs[0, :])
    cols_ret = [f"Return_{h}" for h in PREDICTION_HORIZONS]
    data_melted = df_entries.melt(value_vars=cols_ret, var_name="Horizon", value_name="Return %")
    sns.boxplot(x="Horizon", y="Return %", data=data_melted, ax=ax1, palette="viridis")
    ax1.axhline(0, color='r', linestyle='--')
    ax1.set_title("Distribution of Returns after N Bars (Raw Signal)", fontsize=14)

    # 2. MFE vs MAE Scatter (Risk vs Reward potential)
    # We use the longest horizon for this
    ax2 = fig.add_subplot(gs[1, 0])
    h_max = PREDICTION_HORIZONS[-1]
    sns.scatterplot(x=f"MAE_{h_max}", y=f"MFE_{h_max}", data=df_entries, hue="Macro_Score", ax=ax2, palette="coolwarm")
    ax2.plot([0, -5], [0, 5], color='gray', linestyle='--')  # 1:1 line roughly
    ax2.set_title(f"MFE vs MAE (Horizon: {h_max} bars)", fontsize=12)
    ax2.set_xlabel("Max Adverse Excursion (%) - Pain")
    ax2.set_ylabel("Max Favorable Excursion (%) - Gain")

    # 3. Macro Score Effectiveness
    ax3 = fig.add_subplot(gs[1, 1])
    # Group by Macro Score and get median return at shortest horizon
    h_short = PREDICTION_HORIZONS[1]  # e.g., 4 bars
    sns.barplot(x="Macro_Score", y=f"Return_{h_short}", data=df_entries, ax=ax3, palette="RdBu", errorbar=None)
    ax3.set_title(f"Avg Return after {h_short} bars by Macro Score", fontsize=12)

    # 4. Win Rate over Time
    ax4 = fig.add_subplot(gs[2, :])
    win_rates = []
    for h in PREDICTION_HORIZONS:
        wr = len(df_entries[df_entries[f"Return_{h}"] > 0]) / len(df_entries) * 100
        win_rates.append(wr)

    ax4.plot(PREDICTION_HORIZONS, win_rates, marker='o', linestyle='-', color='green')
    ax4.axhline(50, color='red', linestyle='--')
    ax4.set_ylim(30, 70)
    ax4.set_title("Win Rate Probability over Time Horizons", fontsize=12)
    ax4.set_xlabel("Bars Held")
    ax4.set_ylabel("Win Rate (%)")

    plt.tight_layout()
    plt.show()

    # --- PRINT SUMMARY STATS ---
    print("\n=== ENTRY SIGNAL ANALYSIS ===")
    print(f"Total Signals: {len(df_entries)}")
    for h in PREDICTION_HORIZONS:
        avg_ret = df_entries[f"Return_{h}"].mean()
        avg_mfe = df_entries[f"MFE_{h}"].mean()
        avg_mae = df_entries[f"MAE_{h}"].mean()
        win_rate = len(df_entries[df_entries[f"Return_{h}"] > 0]) / len(df_entries) * 100

        print(f"\nHorizon {h} Bars:")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Avg Return: {avg_ret:.3f}%")
        print(f"  Avg MFE (Potential Reward): {avg_mfe:.3f}%")
        print(f"  Avg MAE (Potential Risk): {avg_mae:.3f}%")
        print(f"  Expectancy Ratio (MFE/|MAE|): {abs(avg_mfe / avg_mae):.2f}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Macro
    macro_proc = MacroProcessor(FRED_API_KEY)
    df_macro = macro_proc.fetch_and_process()

    # 2. Loop Symbols
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            df_ohlc = load_csv(symbol, tf)
            if df_ohlc.empty: continue

            # Filter Date
            df_ohlc = df_ohlc[(df_ohlc["DateTime"] >= start_date) & (df_ohlc["DateTime"] <= end_date)]

            # Merge
            if not df_macro.empty:
                df_combined = merge_macro_to_ohlc(df_ohlc, df_macro)
            else:
                df_combined = df_ohlc
                df_combined['USD_Score'] = 0

            # Run Entry Analysis
            entries_df = analyze_entries(df_combined, symbol)

            # Visualize
            plot_entry_stats(entries_df)