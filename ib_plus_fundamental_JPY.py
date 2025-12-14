import pandas as pd
import os
import numpy as np
from fredapi import Fred

# --- CONFIGURATION ---
# 1. API KEYS & PATHS
FRED_API_KEY = ''  # <-- INSERT KEY HERE
FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data"

# 2. STRATEGY SETTINGS
SYMBOLS = ["USDJPY"]  # Focused on Gold for this example logic
TIMEFRAMES = ["1h"]
RISK_REWARD = 2.0
INITIAL_CAPITAL = 10000.0
RISK_PERCENT = 1.0

# 3. DATE RANGE
start_date = pd.to_datetime("2010-09-20")
end_date = pd.to_datetime("2014-09-20")


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
            'NAPM': 'ISM_PMI'
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
        df_macro['Yield_Change'] = df_macro['US_Treasury_2Y_Yield'].diff()
        df_macro.loc[df_macro['Yield_Change'] > 0, 'USD_Score'] += 2  # Weighted heavier
        df_macro.loc[df_macro['Yield_Change'] < 0, 'USD_Score'] -= 2

        # Rule 3: PMI
        #df_macro.loc[df_macro['ISM_PMI'] > 50, 'USD_Score'] += 1
        #df_macro.loc[df_macro['ISM_PMI'] < 50, 'USD_Score'] -= 1

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

            if symbol == "XAUUSD":
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