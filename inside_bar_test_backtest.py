import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
SYMBOLS = ["USDJPY", "XAUUSD", "BRENTCMDUSD"]
TIMEFRAMES = ["1H", "4H", "D"]
FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data"
RISK_REWARD = 2.0
INITIAL_CAPITAL = 10000.0
RISK_PERCENT = 1.0

# --- TIME SETTINGS ---
StartHour = 7
StartMinute = 0
EndHour = 17
EndMinute = 30

# --- DATE RANGE ---
start_date = pd.to_datetime("2010-09-20")
end_date = pd.to_datetime("2014-09-20")


# --- LOAD DATA FUNCTION ---
def load_csv(symbol, timeframe):
    filepath = os.path.join(FOLDER, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        # Assuming MT5 Export format (Date, Open, High, Low, Close, Vol)
        df = pd.read_csv(filepath, sep="\t", header=None,
                         names=["DateTime", "Open", "High", "Low", "Close", "Minutes"])
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        return df
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
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
    ticket_counter = 100000  # Fake ticket number start

    i = 1
    n = len(df)

    while i < n - 1:

        bar_time = df.loc[i, "DateTime"]
        if timeframe != "D":
            times = df["DateTime"].dt
            # Construct the mask logic
            time_mask = (
                    ((times.hour > StartHour) & (times.hour < EndHour)) |
                    ((times.hour == StartHour) & (times.minute >= StartMinute)) |
                    ((times.hour == EndHour) & (times.minute <= EndMinute))
            )
        else:
            time_mask = pd.Series(True, index=df.index)

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
                # Formula: Risk Money / Distance
                risk_money = current_balance * (RISK_PERCENT / 100.0)

                # Note: In real Forex, we need TickValue.
                # Here we simplify: units = money / price_distance
                units = risk_money / risk_dist
                lots = units / 100000.0  # Standard Lot = 100k units

                if direction == "Buy":
                    take_profit = entry_price + (risk_dist * RISK_REWARD)
                else:
                    take_profit = entry_price - (risk_dist * RISK_REWARD)

                # 2. Find Exit (Loop forward)
                outcome = ""
                exit_price = 0.0
                exit_time = None

                for j in range(i, n):
                    bar_low = df.loc[j, "Low"]
                    bar_high = df.loc[j, "High"]
                    bar_time = df.loc[j, "DateTime"]

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

                # 3. Calculate Profit & Balance
                if outcome != "":
                    # Gross Profit = (Price Diff) * Units
                    if direction == "Buy":
                        gross_profit = (exit_price - entry_price) * units
                    else:
                        gross_profit = (entry_price - exit_price) * units

                    # Update Balance
                    current_balance += gross_profit
                    ticket_counter += 1

                    # 4. Record Trade (MT5 Format)
                    trade_record = {
                        "Ticket": ticket_counter,
                        "OpenTime": df.loc[i, "DateTime"],  # Breakout time
                        "Type": direction,  # "Buy" or "Sell"
                        "Lots": round(lots, 2),
                        "OpenPrice": round(entry_price, 5),
                        "ClosePrice": round(exit_price, 5),
                        "Commission": 0.0,  # Simplified
                        "Swap": 0.0,  # Simplified
                        "Profit": round(gross_profit, 2),
                        "Comment": "Inside Bar Strategy",
                        "Balance": round(current_balance, 2)
                    }
                    trades.append(trade_record)

                    # Jump forward to avoid overlapping trades
                    i = j

        i += 1
    df_backtest_trades = pd.DataFrame(trades)

    # --- Summary statistics ---
    total_trades = len(df_backtest_trades)
    wins = len(df_backtest_trades[df_backtest_trades["Profit"] > 0])
    losses = len(df_backtest_trades[df_backtest_trades["Profit"] < 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    final_balance = current_balance
    max_drawdown = (INITIAL_CAPITAL - np.min(df_backtest_trades["Balance"]))/100
    daily_returns = df_backtest_trades['Balance'].pct_change()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    print(
        f"{symbol} Summary: Total Trades={total_trades}, "
        f"Wins={wins}, "
        f"Losses={losses}, "
        f"Win Rate={win_rate:.2f}%, "
        f"Sharpe Ratio={sharpe_ratio:.2f},"
        f"Final Balance=${final_balance:.2f},"
        f" Max Drawdown=-{max_drawdown:.2f}%")

    return df_backtest_trades


# --- SAVE TRADES TO CSV FUNCTION ---
def save_trades_to_csv(df_trades, symbol, timeframe, output_dir):
    """
    Save trades DataFrame to CSV with MT5-like columns.
    """
    if df_trades.empty:
        print(f"No trades to save for {symbol} {timeframe}.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{symbol}_{timeframe}_Report.csv"
    path = os.path.join(output_dir, filename)

    cols = ["Ticket", "OpenTime", "Type", "Lots", "OpenPrice",
            "ClosePrice", "Commission", "Swap", "Profit",
            "Comment", "Balance"]

    df_trades[cols].to_csv(path, index=False)
    print(f"Saved {len(df_trades)} trades to {path}")
    print(f"Final Balance: ${df_trades.iloc[-1]['Balance']}")


# --- MAIN BLOCK ---
if __name__ == "__main__":
    output_dir = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_trades"

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"Processing {symbol} {tf}...")


            df_data = load_csv(symbol, tf)

            # date filter
            df_data = df_data[(df_data["DateTime"] >= start_date) &
                              (df_data["DateTime"] <= end_date)].reset_index(drop=True)

            if df_data.empty:
                print(f"No data in date range {start_date} → {end_date}")
                print("No data found.")
                print("-" * 30)
                continue

            df_trades = run_strategy_detailed(df_data, symbol, tf)

            #save_trades_to_csv(df_trades, symbol, tf, output_dir)
            print("-" * 30)
