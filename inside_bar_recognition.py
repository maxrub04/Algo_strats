import pandas as pd
import os

symbols = ["EURUSD", "USDJPY", "GBPUSD", "XAGUSD","XAUUSD","BRENTCMDUSD"]
timeframes = ["1H", "4H", "D"]  # 1H, 4H, Daily
folder = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/data"  # CSV folder


def load_csv(symbol, timeframe):
    filepath = os.path.join(folder, f"{symbol}_{timeframe}.csv")
    # CSV without header
    df = pd.read_csv(filepath, header=None, sep="\t",
                     names=["DateTime", "Open", "High", "Low", "Close", "Volume"])
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    # Применяем фильтр по времени только для часовых таймфреймов
    if timeframe != "D":
        df["Time"] = df["DateTime"].dt.time
        mask = ((df["DateTime"].dt.time >= pd.to_datetime("07:00").time()) &
                (df["DateTime"].dt.time <= pd.to_datetime("17:30").time()))
        df = df[mask]

    df = df[["DateTime", "Open", "High", "Low", "Close"]]
    return df


def detect_inside_bars(df):
    df = df.copy()
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])
    return df[df["Inside_Bar"]]


results = {}

for symbol in symbols:
    results[symbol] = {}
    for tf in timeframes:
        df = load_csv(symbol, tf)
        inside_bars = detect_inside_bars(df)
        results[symbol][tf] = inside_bars

# Print last 5 inside bars for each symbol and timeframe
for symbol in symbols:
    print(f"=== {symbol} ===")
    for tf in timeframes:
        print(f"{tf} timeframe:")
        print(results[symbol][tf].tail())
        print()