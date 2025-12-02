import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta  # Recommended for easy indicators, or we can calculate manually

# --- CONFIGURATION ---
SYMBOLS = ["JPY=X", "GC=F", "BZ=F"]  # Yahoo Tickers: USDJPY, Gold, Brent
TIMEFRAMES = ["30m","1h","1d"]  # Yahoo supports: 15m, 30m, 1h, 1d
RISK_REWARD = 2.0
INITIAL_CAPITAL = 10000.0
RISK_PERCENT = 1.0

# --- STRATEGY SETTINGS ---
VOL_MA_PERIOD = 20  # Moving Average for Volume
VOL_MULTIPLIER = 1.2  # Breakout volume must be 20% higher than average
ORDER_EXPIRATION_BARS = 5  # How many bars to wait for the pullback before cancelling

# --- TIME SETTINGS (UTC implied by Yahoo usually, adjust as needed) ---
StartHour = 8
StartMinute = 0
EndHour = 20
EndMinute = 0

# --- DATE RANGE ---
start_date = "2023-01-01"
end_date = "2024-01-01"


# --- LOAD DATA FUNCTION (YAHOO FINANCE) ---
def load_data_yahoo(symbol, interval):
    print(f"Downloading {symbol} from Yahoo Finance...")
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (common issue with new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        # Ensure standard column names
        df.rename(columns={"Date": "DateTime", "Datetime": "DateTime"}, inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return pd.DataFrame()


# --- MAIN STRATEGY LOGIC ---
def run_strategy_detailed(df, symbol, timeframe):
    trades = []

    # 1. Pre-calculate Indicators
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)

    # Inside Bar Logic
    df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

    # Volume Filter: Simple Moving Average of Volume
    df["Vol_MA"] = df["Volume"].rolling(window=VOL_MA_PERIOD).mean()

    current_balance = INITIAL_CAPITAL
    ticket_counter = 100000

    i = 2  # Start from 2 to have enough history for MA
    n = len(df)

    while i < n - 1:
        # Time Filter Check
        bar_time = df.loc[i, "DateTime"]
        if timeframe != "1d":
            # Simple check if bar hour is within trading window
            if not (StartHour <= bar_time.hour < EndHour):
                i += 1
                continue

        # --- PATTERN RECOGNITION ---
        # Look at bar (i-1) for Inside Bar pattern
        # Look at bar (i) for the BREAKOUT with VOLUME

        is_ib = df.loc[i - 1, "Is_Inside_Bar"]

        if is_ib:
            ib_high = df.loc[i - 1, "High"]
            ib_low = df.loc[i - 1, "Low"]

            curr_close = df.loc[i, "Close"]
            curr_vol = df.loc[i, "Volume"]
            avg_vol = df.loc[i, "Vol_MA"]

            # --- SIGNAL GENERATION ---
            signal_dir = ""
            limit_price = 0.0
            stop_loss = 0.0

            # 1. Bullish Breakout with Volume
            # Price closed above mother high AND Volume is significant
            if (curr_close > ib_high) and (curr_vol > avg_vol * VOL_MULTIPLIER):
                signal_dir = "Buy"
                limit_price = ib_high  # We wait for price to drop back to this level
                stop_loss = ib_low  # SL below mother bar

            # 2. Bearish Breakout with Volume
            # Price closed below mother low AND Volume is significant
            elif (curr_close < ib_low) and (curr_vol > avg_vol * VOL_MULTIPLIER):
                signal_dir = "Sell"
                limit_price = ib_low  # We wait for price to rise back to this level
                stop_loss = ib_high  # SL above mother bar

            # --- ORDER MANAGEMENT (WAIT FOR FILL) ---
            if signal_dir != "":
                risk_dist = abs(limit_price - stop_loss)
                if risk_dist == 0:
                    i += 1
                    continue

                # Setup TP
                take_profit = 0.0
                if signal_dir == "Buy":
                    take_profit = limit_price + (risk_dist * RISK_REWARD)
                else:
                    take_profit = limit_price - (risk_dist * RISK_REWARD)

                # --- PENDING ORDER LOOP ---
                # We do not enter at 'i'. We wait for subsequent bars (j) to touch limit_price
                is_filled = False
                fill_time = None

                # Check next X bars for entry (Expiration logic)
                # We start checking from i+1 because i is the breakout bar itself
                search_end = min(i + 1 + ORDER_EXPIRATION_BARS, n)

                fill_index = -1

                for j in range(i + 1, search_end):
                    future_low = df.loc[j, "Low"]
                    future_high = df.loc[j, "High"]

                    # Check for Fill
                    if signal_dir == "Buy":
                        # To fill a Buy Limit, price must drop to or below limit_price
                        if future_low <= limit_price:
                            is_filled = True
                            fill_time = df.loc[j, "DateTime"]
                            fill_index = j
                            break

                    elif signal_dir == "Sell":
                        # To fill a Sell Limit, price must rise to or above limit_price
                        if future_high >= limit_price:
                            is_filled = True
                            fill_time = df.loc[j, "DateTime"]
                            fill_index = j
                            break

                    # Cancel order if price runs away too far without filling? (Optional)
                    # For now, we just rely on ORDER_EXPIRATION_BARS

                # --- TRADE MANAGEMENT (IF FILLED) ---
                if is_filled:

                    # Calculate position size based on current balance
                    risk_money = current_balance * (RISK_PERCENT / 100.0)
                    units = risk_money / risk_dist
                    lots = units / 100000.0

                    outcome = ""
                    exit_price = 0.0

                    # Loop from the fill candle onwards to find exit
                    for k in range(fill_index, n):
                        bar_low = df.loc[k, "Low"]
                        bar_high = df.loc[k, "High"]

                        if signal_dir == "Buy":
                            if bar_low <= stop_loss:
                                exit_price = stop_loss
                                outcome = "Loss"
                                break
                            elif bar_high >= take_profit:
                                exit_price = take_profit
                                outcome = "Win"
                                break

                        elif signal_dir == "Sell":
                            if bar_high >= stop_loss:
                                exit_price = stop_loss
                                outcome = "Loss"
                                break
                            elif bar_low <= take_profit:
                                exit_price = take_profit
                                outcome = "Win"
                                break

                    # Record the trade if it closed
                    if outcome != "":
                        gross_profit = 0.0
                        if signal_dir == "Buy":
                            gross_profit = (exit_price - limit_price) * units
                        else:
                            gross_profit = (limit_price - exit_price) * units

                        current_balance += gross_profit
                        ticket_counter += 1

                        trade_record = {
                            "Ticket": ticket_counter,
                            "OpenTime": fill_time,
                            "Type": f"{signal_dir} Limit",
                            "Lots": round(lots, 2),
                            "OpenPrice": round(limit_price, 5),
                            "ClosePrice": round(exit_price, 5),
                            "Profit": round(gross_profit, 2),
                            "Comment": f"Retest Vol Breakout ({outcome})",
                            "Balance": round(current_balance, 2)
                        }
                        trades.append(trade_record)

                        # Move main loop 'i' to 'k' to avoid overlapping trades
                        i = k
                else:
                    # Order expired without fill
                    # print(f"Order Expired for {symbol} at {df.loc[i, 'DateTime']}")
                    pass

        i += 1

    df_trades = pd.DataFrame(trades)

    # Calculate Stats
    if not df_trades.empty:
        wins = len(df_trades[df_trades["Profit"] > 0])
        total = len(df_trades)
        wr = (wins / total) * 100
        dd = (INITIAL_CAPITAL - df_trades["Balance"].min()) / INITIAL_CAPITAL * 100 if df_trades[
                                                                                           "Balance"].min() < INITIAL_CAPITAL else 0

        print(
            f"[{symbol} {timeframe}] Trades: {total} | Win Rate: {wr:.1f}% | Balance: ${current_balance:.0f} | DD: {dd:.1f}%")
    else:
        print(f"[{symbol} {timeframe}] No trades executed.")

    return df_trades


# --- MAIN BLOCK ---
if __name__ == "__main__":
    output_dir = "backtest_results"  # Folder for results

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            # 1. Load Data
            df_data = load_data_yahoo(symbol, tf)

            if df_data.empty or len(df_data) < 50:
                continue

            # 2. Run Strategy
            df_trades = run_strategy_detailed(df_data, symbol, tf)

            # 3. Save Logic (Optional)
            if not df_trades.empty:
                import os

                if not os.path.exists(output_dir): os.makedirs(output_dir)
                df_trades.to_csv(f"{output_dir}/{symbol}_{tf}_limit.csv", index=False)