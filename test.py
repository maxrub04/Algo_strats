import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
prices = np.random.normal(loc=35000, scale=500, size=30)
df = pd.DataFrame({"Date": dates, "Price": prices})
df.set_index("Date", inplace=True)

def compute_rsi(series, window=14):
    """
    Calculates the RSI for the given price series.
    Arguments:
    series (pd.Series): The price series.
    window (int): RSI calculation period.
    Returns:
    pd.Series: RSI values.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    # Calculate RS
    rs = avg_gain / avg_loss
    # rsi
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI_14"] = compute_rsi(df["Price"], window=14)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["RSI_14"], label="RSI (14)", color="purple")
plt.axhline(70, color="red", linestyle="--", label="Overbought level")
plt.axhline(30, color="green", linestyle="--", label="Oversold level")
plt.title("Relative Strength Index (RSI)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()