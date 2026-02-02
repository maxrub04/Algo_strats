import pandas as pd
import os
import numpy as np
from numba import njit, float64, int64, int32, boolean

# --- CONFIGURATION ---
SYMBOLS = ["USDJPY", "XAUUSD", "BRENTCMDUSD"]
TIMEFRAMES = ["1H", "4H", "D"]
FOLDER = ""  # Update path if needed
OUTPUT_DIR = ""

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


def load_csv(symbol, timeframe):
    filepath = os.path.join(FOLDER, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(filepath):
        # Fallback for testing or wrong path
        return pd.DataFrame()

    try:
        # Assuming MT5 Export format
        df = pd.read_csv(filepath, sep="\t", header=None,
                         names=["DateTime", "Open", "High", "Low", "Close", "Vol"])
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        return df
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# NUMBA CORE LOGIC (The Engine)
# This function runs at C++ speed because of @njit
# -----------------------------------------------------------------------------
@njit
def run_backtest_numba(times_h, times_m, highs, lows, is_inside_bar,
                       risk_reward, risk_percent, initial_capital,
                       start_h, start_m, end_h, end_m, is_daily):
    n = len(highs)

    # Lists to store trade results (using simple arrays logic for Numba compatibility)
    # 0: Ticket, 1: Type(1/-1), 2: Lots, 3: OpenPrice, 4: ClosePrice, 5: Profit, 6: Balance, 7: OpenIndex, 8: CloseIndex

    # We estimate max trades to pre-allocate or use lists.
    # For simplicity in Numba, we simulate lists.
    tickets = []
    types = []  # 1 for Buy, -1 for Sell
    lots_list = []
    open_prices = []
    close_prices = []
    profits = []
    balances = []
    open_indices = []  # To map back to time later

    current_balance = initial_capital
    ticket_counter = 100000

    i = 1
    # Main Loop
    while i < n - 1:

        # 1. Time Filter
        if not is_daily:
            h = times_h[i]
            m = times_m[i]

            # Logic: Outside trading hours? Skip.
            is_valid_time = False
            if (h > start_h) and (h < end_h):
                is_valid_time = True
            elif (h == start_h) and (m >= start_m):
                is_valid_time = True
            elif (h == end_h) and (m <= end_m):
                is_valid_time = True

            if not is_valid_time:
                i += 1
                continue

        # 2. Check Signal (Inside Bar on Previous Candle)
        # Note: Your logic checks if i-1 was an IB.
        if is_inside_bar[i - 1]:

            # The "Inside Bar" (The small one)
            ib_high = highs[i - 1]
            ib_low = lows[i - 1]

            # The Current Candle (Potential Breakout)
            curr_high = highs[i]
            curr_low = lows[i]

            direction = 0  # 0: None, 1: Buy, -1: Sell
            entry_price = 0.0
            stop_loss = 0.0

            # --- ENTRY TRIGGERS ---
            if curr_high > ib_high:
                direction = 1  # Buy
                entry_price = ib_high
                stop_loss = ib_low
            elif curr_low < ib_low:
                direction = -1  # Sell
                entry_price = ib_low
                stop_loss = ib_high

            # --- EXECUTION ---
            if direction != 0:
                risk_dist = abs(entry_price - stop_loss)

                if risk_dist > 0.00001:  # Avoid zero division

                    # Position Size
                    risk_money = current_balance * (risk_percent / 100.0)
                    units = risk_money / risk_dist
                    lots = units / 100000.0

                    take_profit = 0.0
                    if direction == 1:
                        take_profit = entry_price + (risk_dist * risk_reward)
                    else:
                        take_profit = entry_price - (risk_dist * risk_reward)

                    # --- EXIT LOOP (Fast Forward) ---
                    outcome = 0  # 0: Running, 1: Win, -1: Loss
                    exit_price = 0.0

                    # We start looking from current candle 'i' (intra-bar) or next ones
                    # Your original code loops from 'i' to 'n'

                    jump_to_index = -1

                    for j in range(i, n):
                        j_low = lows[j]
                        j_high = highs[j]

                        if direction == 1:  # Long
                            if j_low <= stop_loss:
                                exit_price = stop_loss
                                outcome = -1
                                jump_to_index = j
                                break
                            elif j_high >= take_profit:
                                exit_price = take_profit
                                outcome = 1
                                jump_to_index = j
                                break
                        else:  # Short
                            if j_high >= stop_loss:
                                exit_price = stop_loss
                                outcome = -1
                                jump_to_index = j
                                break
                            elif j_low <= take_profit:
                                exit_price = take_profit
                                outcome = 1
                                jump_to_index = j
                                break

                    # If trade closed
                    if outcome != 0:
                        gross_profit = 0.0
                        if direction == 1:
                            gross_profit = (exit_price - entry_price) * units
                        else:
                            gross_profit = (entry_price - exit_price) * units

                        current_balance += gross_profit
                        ticket_counter += 1

                        # Store Data
                        tickets.append(ticket_counter)
                        types.append(direction)
                        lots_list.append(lots)
                        open_prices.append(entry_price)
                        close_prices.append(exit_price)
                        profits.append(gross_profit)
                        balances.append(current_balance)
                        open_indices.append(i)  # Time of entry trigger

                        # Move the main iterator forward to avoid overlapping trades
                        i = jump_to_index

        i += 1

    return tickets, types, lots_list, open_prices, close_prices, profits, balances, open_indices


# -----------------------------------------------------------------------------
# WRAPPER FUNCTION
# Prepares data for Numba and reconstructs DataFrame
# -----------------------------------------------------------------------------
def run_strategy_vectorized(df, symbol, timeframe):
    # 1. Pre-calculate Indicators (Vectorized in Pandas)
    # Logic: Inside Bar means High < PrevHigh AND Low > PrevLow
    # Note: We shift(1) to compare current row with previous row
    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)

    # Boolean mask for Inside Bar
    is_ib_series = (df["High"] < prev_high) & (df["Low"] > prev_low)

    # 2. Prepare NumPy arrays for Numba
    # Numba hates DataFrames, loves Arrays
    highs = df["High"].values.astype(np.float64)
    lows = df["Low"].values.astype(np.float64)
    times_h = df["DateTime"].dt.hour.values.astype(np.int64)
    times_m = df["DateTime"].dt.minute.values.astype(np.int64)
    is_ib_arr = is_ib_series.values  # Boolean array

    is_daily = (timeframe == "D")

    # 3. Call Numba Function
    results = run_backtest_numba(
        times_h, times_m, highs, lows, is_ib_arr,
        RISK_REWARD, RISK_PERCENT, INITIAL_CAPITAL,
        StartHour, StartMinute, EndHour, EndMinute, is_daily
    )

    # 4. Unpack Results
    tickets, types, lots, opens, closes, profits, balances, open_idx = results

    if len(tickets) == 0:
        return pd.DataFrame()

    # 5. Reconstruct DataFrame
    df_trades = pd.DataFrame({
        "Ticket": tickets,
        "OpenTime": df.iloc[open_idx]["DateTime"].values,  # Map indices back to timestamps
        "Type": ["Buy" if t == 1 else "Sell" for t in types],
        "Lots": np.round(lots, 2),
        "OpenPrice": np.round(opens, 5),
        "ClosePrice": np.round(closes, 5),
        "Commission": 0.0,
        "Swap": 0.0,
        "Profit": np.round(profits, 2),
        "Comment": "Inside Bar Numba",
        "Balance": np.round(balances, 2)
    })

    # --- STATISTICS ---
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades["Profit"] > 0])
    losses = len(df_trades[df_trades["Profit"] < 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Drawdown Calc
    running_max = df_trades["Balance"].cummax()
    drawdown = (df_trades["Balance"] - running_max)
    max_dd_money = drawdown.min()
    max_dd_pct = (drawdown / running_max).min() * 100

    # Simple Sharpe (Per trade approx)
    avg_return = df_trades["Profit"].mean()
    std_return = df_trades["Profit"].std()
    sharpe = (avg_return / std_return) if std_return != 0 else 0
    # Note: To get annual sharpe, we need daily returns, not trade returns.
    # Keeping it simple here as per original logic.

    print(f"{symbol} {timeframe} Results:")
    print(f"  Trades: {total_trades} | WinRate: {win_rate:.2f}%")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Final Balance: ${df_trades.iloc[-1]['Balance']:.2f}")
    print(f"  Max Drawdown: {max_dd_pct:.2f}% (${max_dd_money:.2f})")
    print("-" * 30)

    return df_trades


def save_trades_to_csv(df_trades, symbol, timeframe, output_dir):
    if df_trades.empty:
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{symbol}_{timeframe}_Report.csv"
    path = os.path.join(output_dir, filename)
    df_trades.to_csv(path, index=False)
    print(f"Saved to {path}")


# --- MAIN BLOCK ---
if __name__ == "__main__":

    # Just to warm up Numba (First run is always slow due to compilation)
    print("Compiling Numba Strategy...")
    dummy_arr = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    dummy_bool = np.array([False, True, False], dtype=bool)
    dummy_int = np.array([1, 2, 3], dtype=np.int64)
    try:
        run_backtest_numba(dummy_int, dummy_int, dummy_arr, dummy_arr, dummy_bool,
                           2.0, 1.0, 10000.0, 10, 0, 18, 0, False)
        print("Compilation Complete. Starting Backtest.\n")
    except Exception as e:
        print(f"Compilation Warning (Ignorable): {e}")

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:

            df_data = load_csv(symbol, tf)

            # Apply Date Filter
            if not df_data.empty:
                df_data = df_data[(df_data["DateTime"] >= start_date) &
                                  (df_data["DateTime"] <= end_date)].reset_index(drop=True)

            if df_data.empty or len(df_data) < 5:
                print(f"No data for {symbol} {tf}")
                continue

            df_res = run_strategy_vectorized(df_data, symbol, tf)

            """if not df_res.empty:
                save_trades_to_csv(df_res, symbol, tf, OUTPUT_DIR)"""
