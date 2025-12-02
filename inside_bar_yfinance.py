import pandas as pd
import numpy as np
import yfinance as yf
import os


SYMBOLS = ["JPY=X", "GC=F", "BZ=F"]
TIMEFRAMES = ["1h", "4h", "1d"]
RISK_REWARD = 2.0
INITIAL_CAPITAL = 10000.0
RISK_PERCENT = 1.0

# --- TIME SETTINGS ---
# Strategy active trading hours
StartHour = 7
StartMinute = 0
EndHour = 17
EndMinute = 30

# --- DATE RANGE ---
# NOTE: yfinance restriction: 1H data is only available for the last 730 days.
# Ensure these dates are within the last 2 years for 1H/4H testing.
start_date = pd.to_datetime("2023-06-01")
end_date = pd.to_datetime("2024-06-01")


# --- DATA PROCESSING FUNCTIONS ---

def resample_to_4h(df):
    """
    Aggregates 1H data into 4H candles.
    """
    # Set DateTime as index for resampling
    df = df.set_index("DateTime")

    # Define aggregation rules
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }

    # Resample to 4 Hours
    df_resampled = df.resample("4h").agg(agg_dict).dropna()

    # Reset index to make DateTime a column again
    return df_resampled.reset_index()


def download_data(symbol, timeframe, start, end):
    """
    Downloads data from yfinance and prepares it for the strategy.
    """
    # Determine interval for yfinance
    yf_interval = "1d"
    if timeframe in ["1H", "4H"]:
        yf_interval = "1h"

    print(f"Downloading {symbol} ({timeframe}) via yfinance...")

    try:
        df = yf.download(symbol, start=start, end=end, interval=yf_interval, progress=False)

        if df.empty:
            print(f"Warning: No data found for {symbol}. Check dates or ticker.")
            return pd.DataFrame()

        # Handle MultiIndex columns (common in new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index to get Date/Datetime as a column
        df = df.reset_index()

        # Normalize column names
        df.rename(columns={"Date": "DateTime", "Datetime": "DateTime"}, inplace=True)

        # Remove Timezone information (make naive) to avoid comparison errors
        if pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
            df["DateTime"] = df["DateTime"].dt.tz_localize(None)

        # Keep only necessary columns
        req_cols = ["DateTime", "Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in req_cols):
            print(f"Missing columns. Got: {df.columns}")
            return pd.DataFrame()

        df = df[req_cols].copy()

        # Handle 4H Resampling
        if timeframe == "4H":
            if len(df) < 4:
                print("Not enough 1H data to resample to 4H.")
                return pd.DataFrame()
            df = resample_to_4h(df)

        return df

    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


# --- MAIN STRATEGY LOGIC ---
def run_strategy_detailed(df, symbol, timeframe):
    """
    Simulates trading and returns a DataFrame with columns matching MT5 report.
    """
    trades = []

    # Pre-calculate indicators
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

    current_balance = INITIAL_CAPITAL
    ticket_counter = 100000

    i = 1
    n = len(df)

    while i < n - 1:
        bar_time = df.loc[i, "DateTime"]

        # Time Filter logic
        if timeframe != "D":
            current_hour = bar_time.hour
            current_minute = bar_time.minute

            # Simple time check
            is_within_time = False
            if StartHour < current_hour < EndHour:
                is_within_time = True
            elif current_hour == StartHour and current_minute >= StartMinute:
                is_within_time = True
            elif current_hour == EndHour and current_minute <= EndMinute:
                is_within_time = True

            if not is_within_time:
                i += 1
                continue

        # Check Inside Bar on previous candle (i-1)
        if df.loc[i - 1, "Is_Inside_Bar"]:
            # mother candle
            ib_high = df.loc[i - 1, "High"]
            ib_low = df.loc[i - 1, "Low"]

            # Current candle (i) checks for breakout
            curr_high = df.loc[i, "High"]
            curr_low = df.loc[i, "Low"]

            entry_price = 0.0
            stop_loss = 0.0
            take_profit = 0.0
            direction = ""  # "Buy" or "Sell"

            # --- ENTRY TRIGGERS ---
            if curr_high > ib_high:
                direction = "Buy"
                entry_price = ib_high
                stop_loss = ib_low

            elif curr_low < ib_low:
                direction = "Sell"
                entry_price = ib_low
                stop_loss = ib_high

            # --- TRADE EXECUTION ---
            if direction != "":
                risk_dist = abs(entry_price - stop_loss)

                # Avoid zero division error
                if risk_dist == 0:
                    i += 1
                    continue

                # 1. Calculate Position Size (Lots)
                risk_money = current_balance * (RISK_PERCENT / 100.0)
                units = risk_money / risk_dist
                lots = units / 100000.0

                if direction == "Buy":
                    take_profit = entry_price + (risk_dist * RISK_REWARD)
                else:
                    take_profit = entry_price - (risk_dist * RISK_REWARD)

                # 2. Find Exit (Loop forward)
                outcome = ""
                exit_price = 0.0

                # Start looking from the current bar (i) or next bar?
                # Usually breakout happens inside bar 'i', so we check exit from 'i' onwards
                for j in range(i, n):
                    bar_low = df.loc[j, "Low"]
                    bar_high = df.loc[j, "High"]

                    if direction == "Buy":
                        if bar_low <= stop_loss:  # Hit SL
                            exit_price = stop_loss
                            outcome = "Loss"
                            break
                        elif bar_high >= take_profit:  # Hit TP
                            exit_price = take_profit
                            outcome = "Win"
                            break

                    elif direction == "Sell":
                        if bar_high >= stop_loss:  # Hit SL
                            exit_price = stop_loss
                            outcome = "Loss"
                            break
                        elif bar_low <= take_profit:  # Hit TP
                            exit_price = take_profit
                            outcome = "Win"
                            break

                    # Force close at end of data if no TP/SL hit
                    if j == n - 1:
                        exit_price = df.loc[j, "Close"]
                        outcome = "End of Data"

                # 3. Calculate Profit & Balance
                if outcome != "":
                    if direction == "Buy":
                        gross_profit = (exit_price - entry_price) * units
                    else:
                        gross_profit = (entry_price - exit_price) * units

                    current_balance += gross_profit
                    ticket_counter += 1

                    trade_record = {
                        "Ticket": ticket_counter,
                        "OpenTime": df.loc[i, "DateTime"],
                        "Type": direction,
                        "Lots": round(lots, 2),
                        "OpenPrice": round(entry_price, 5),
                        "ClosePrice": round(exit_price, 5),
                        "Profit": round(gross_profit, 2),
                        "Balance": round(current_balance, 2),
                        "Outcome": outcome
                    }
                    trades.append(trade_record)

                    # Jump forward to where the trade closed to avoid overlapping trades
                    i = j

        i += 1

    df_backtest_trades = pd.DataFrame(trades)

    # --- Summary statistics ---
    if df_backtest_trades.empty:
        print(f"{symbol} ({timeframe}): No trades generated.")
        return df_backtest_trades

    total_trades = len(df_backtest_trades)
    wins = len(df_backtest_trades[df_backtest_trades["Profit"] > 0])
    losses = len(df_backtest_trades[df_backtest_trades["Profit"] < 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    final_balance = current_balance
    sharpe_ratio = (final_balance / INITIAL_CAPITAL) * np.sqrt(252)

    # Simple Drawdown calculation
    df_backtest_trades["Peak"] = df_backtest_trades["Balance"].cummax()
    df_backtest_trades["Drawdown"] = (df_backtest_trades["Balance"] - df_backtest_trades["Peak"]) / df_backtest_trades[
        "Peak"] * 100
    max_drawdown = df_backtest_trades["Drawdown"].min()

    print(f"--- RESULTS: {symbol} [{timeframe}] ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}% ({wins} W / {losses} L)")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Final Bal:    ${final_balance:.2f}")
    print(f"Max DD:       {max_drawdown:.2f}%")
    print("-" * 30)

    return df_backtest_trades


# --- MAIN BLOCK ---
if __name__ == "__main__":

    # Directory to save results (optional)
    output_dir = "backtest_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:

            # 1. Download & Prepare Data
            df_data = download_data(symbol, tf, start_date, end_date)

            if df_data.empty:
                continue

            # 2. Run Strategy
            df_trades = run_strategy_detailed(df_data, symbol, tf)

            # 3. Save to CSV (Optional)
            """if not df_trades.empty:
                safe_symbol = symbol.replace("=", "").replace("^", "")  # Clean filename
                filename = f"{safe_symbol}_{tf}_Report.csv"
                df_trades.to_csv(os.path.join(output_dir, filename), index=False)"""