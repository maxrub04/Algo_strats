import pandas as pd
import os
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# --- CONFIGURATION ---
FRED_API_KEY = 'bb21719b26aff61c740702ea701a07ed'
FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data"
SYMBOL = "BRENTCMDUSD"
TIMEFRAME = "4h"
INITIAL_CAPITAL = 10000.0
RISK_PERCENT = 1.0

# --- WFA SETTINGS ---
# We will split data into 5 windows
# Each window: 2 Years Optimization -> 1 Year Forward Test
TRAIN_YEARS = 4
TEST_YEARS = 1

# Parameters to Optimize
PARAM_GRID = {
    'risk_reward': [1.5, 2.0, 2.5, 3.0],
    'macro_threshold': [0, 1, 2]  # 0=Trade all, 1=Stronger signal, 2=Very strong only
}


# --- CLASS DEFINITIONS (Kept from your code) ---
class MacroProcessor:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.indicators = {
            'PAYEMS': 'NFP_Total_Jobs',
            'DGS2': 'US_Treasury_2Y_Yield',
            'DFII10': 'TIPS_10Y_Real_Yield',
            'IMPGS': 'Imports',
            'EXPGS': 'Exports',
            'BOPGSTB': 'Trade_Balance',
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
                pass

        if not data_frames: return pd.DataFrame()

        df = pd.concat(data_frames, axis=1).sort_index().ffill().dropna()
        df['USD_Score'] = 0

        # NFP Rule
        df['NFP_Change'] = df['NFP_Total_Jobs'].diff()
        df.loc[df['NFP_Change'] > 0, 'USD_Score'] += 1
        df.loc[df['NFP_Change'] < 0, 'USD_Score'] -= 1

        # TIPS Rule
        df["TIPS_Diff"] = df["TIPS_10Y_Real_Yield"].diff()
        df.loc[df["TIPS_Diff"] > 0, "USD_Score"] += 1
        df.loc[df["TIPS_Diff"] < 0, "USD_Score"] -= 1

        # Trade Balance Rule
        df["Trade_Diff"] = df["Trade_Balance"].diff()
        df.loc[df["Trade_Diff"] > 0, "USD_Score"] += 1
        df.loc[df["Trade_Diff"] < 0, "USD_Score"] -= 1

        df.index.name = 'DateTime'
        return df.reset_index()[['DateTime', 'USD_Score']]


def load_csv(symbol, timeframe):
    filepath = os.path.join(FOLDER, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(filepath): return pd.DataFrame()
    df = pd.read_csv(filepath, sep="\t", header=None, names=["DateTime", "Open", "High", "Low", "Close", "Vol"])
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df


def merge_macro_to_ohlc(df_ohlc, df_macro):
    df_ohlc = df_ohlc.sort_values("DateTime")
    df_macro = df_macro.sort_values("DateTime")
    df_merged = pd.merge_asof(df_ohlc, df_macro, on="DateTime", direction="backward")
    df_merged['USD_Score'] = df_merged['USD_Score'].fillna(0)
    return df_merged


# --- CORE STRATEGY ENGINE (Dynamic Params) ---
def run_strategy(df, risk_reward, macro_threshold):
    # This version accepts params instead of hardcoded values
    trades = []

    # Pre-calc
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

    current_balance = INITIAL_CAPITAL

    # Loop
    # Vectorized loop is hard for inside bar logic, keeping iterative for accuracy
    # To speed up WFA, we can strip down logic to essentials

    i = 1
    n = len(df)
    while i < n - 1:
        if df.loc[i - 1, "Is_Inside_Bar"]:
            ib_high = df.loc[i - 1, "High"]
            ib_low = df.loc[i - 1, "Low"]

            # Entry
            curr_high = df.loc[i, "High"]
            curr_low = df.loc[i, "Low"]

            direction = ""
            if curr_high > ib_high:
                direction = "Buy"; entry = ib_high; sl = ib_low
            elif curr_low < ib_low:
                direction = "Sell"; entry = ib_low; sl = ib_high

            # Filter
            score = df.loc[i, "USD_Score"]
            allowed = False

            # BRENT LOGIC:
            # Sell Brent (Buy USD) if Score >= Threshold
            # Buy Brent (Sell USD) if Score <= -Threshold

            if direction == "Sell" and score >= -macro_threshold: allowed = True  # logic check needed?
            # Let's align with your specific logic:
            # If USD Strong (Score > 0) -> SELL Oil
            # If USD Weak (Score < 0) -> BUY Oil

            if direction == "Sell" and score >= macro_threshold:
                allowed = True  # Strong USD
            elif direction == "Buy" and score <= -macro_threshold:
                allowed = True  # Weak USD

            # Simplify for speed: if macro_threshold is 0, allowed is always True (roughly)
            if macro_threshold == 0: allowed = True

            if direction and allowed:
                dist = abs(entry - sl)
                if dist > 0:
                    units = (current_balance * (RISK_PERCENT / 100)) / dist
                    tp = entry + (dist * risk_reward) if direction == "Buy" else entry - (dist * risk_reward)

                    # Fast Exit Finder
                    exit_price = 0
                    outcome = ""
                    # Slice future data for speed instead of loop
                    subset = df.iloc[i:min(i + 100, n)]  # Look ahead max 100 bars for speed in WFA

                    for idx, row in subset.iterrows():
                        if direction == "Buy":
                            if row['Low'] <= sl: exit_price = sl; outcome = "Loss"; break
                            if row['High'] >= tp: exit_price = tp; outcome = "Win"; break
                        else:
                            if row['High'] >= sl: exit_price = sl; outcome = "Loss"; break
                            if row['Low'] <= tp: exit_price = tp; outcome = "Win"; break

                    if outcome:
                        pnl = (exit_price - entry) * units if direction == "Buy" else (entry - exit_price) * units
                        current_balance += pnl
                        trades.append({'Profit': pnl, 'Date': df.loc[i, 'DateTime']})
                        i = idx  # Jump forward
        i += 1

    if not trades: return 0, 0  # Return 0 profit if no trades

    df_t = pd.DataFrame(trades)
    total_profit = df_t['Profit'].sum()
    return total_profit, df_t


# --- WALK FORWARD ENGINE ---
def run_walk_forward():
    # 1. Prepare Data
    mp = MacroProcessor(FRED_API_KEY)
    df_macro = mp.fetch_and_process()
    df_ohlc = load_csv(SYMBOL, TIMEFRAME)
    df = merge_macro_to_ohlc(df_ohlc, df_macro)

    # 2. Define Windows
    start_dt = df['DateTime'].min()
    end_dt = df['DateTime'].max()

    years = range(start_dt.year, end_dt.year - TEST_YEARS)

    wfa_results = []

    print(f"\nSTARTING WALK FORWARD ANALYSIS ({start_dt.year} - {end_dt.year})")
    print("=" * 60)

    # Iterate through years
    for y in years:
        # Define Train (In-Sample) and Test (Out-of-Sample) Periods
        train_start = pd.Timestamp(f"{y}-01-01")
        train_end = pd.Timestamp(f"{y + TRAIN_YEARS}-01-01")
        test_end = pd.Timestamp(f"{y + TRAIN_YEARS + TEST_YEARS}-01-01")

        if test_end > end_dt: break

        print(
            f"Window: Train [{train_start.date()} -> {train_end.date()}] | Test [{train_end.date()} -> {test_end.date()}]")

        # Slice Data
        df_train = df[(df['DateTime'] >= train_start) & (df['DateTime'] < train_end)].reset_index(drop=True)
        df_test = df[(df['DateTime'] >= train_end) & (df['DateTime'] < test_end)].reset_index(drop=True)

        if df_train.empty or df_test.empty: continue

        # --- OPTIMIZATION STEP (In-Sample) ---
        best_score = -999999
        best_params = None

        # Grid Search
        param_list = list(product(PARAM_GRID['risk_reward'], PARAM_GRID['macro_threshold']))

        for rr, macro in param_list:
            profit, _ = run_strategy(df_train, rr, macro)
            # Fitness function: Simply Total Profit (can be Sharpe or Expectancy)
            if profit > best_score:
                best_score = profit
                best_params = (rr, macro)

        print(
            f"  >> Best Params Found: RR={best_params[0]}, MacroThreshold={best_params[1]} (Profit: ${best_score:.0f})")

        # --- FORWARD TEST STEP (Out-of-Sample) ---
        # Run using the BEST params on NEW data
        test_profit, df_test_trades = run_strategy(df_test, best_params[0], best_params[1])

        print(f"  >> Forward Test Result: ${test_profit:.2f}")

        if not df_test_trades.empty:  # Only save if trades occurred
            if isinstance(df_test_trades, tuple): df_test_trades = df_test_trades[1]  # Handle edge case
            wfa_results.append(df_test_trades)

    # --- AGGREGATE RESULTS ---
    print("=" * 60)
    if wfa_results:
        all_wfa_trades = pd.concat(wfa_results).sort_values('Date').reset_index(drop=True)
        output_filename = f"wfa_results_{SYMBOL}_{TIMEFRAME}_trades.csv"
        all_wfa_trades.to_csv(os.path.join("/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results",
                                    output_filename), index=False)

        # Calculate WFA Equity Curve
        all_wfa_trades['Balance'] = INITIAL_CAPITAL + all_wfa_trades['Profit'].cumsum()

        final_return = ((all_wfa_trades['Balance'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        print(f"WFA Final Return: {final_return:.2f}%")
        print(f"Total Trades: {len(all_wfa_trades)}")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(all_wfa_trades['Date'], all_wfa_trades['Balance'], color='blue', label='Walk-Forward Equity')
        plt.title(f"Walk-Forward Analysis: {SYMBOL} (Rolling Optimization)")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("No trades generated in Walk-Forward Analysis.")


if __name__ == "__main__":
    run_walk_forward()