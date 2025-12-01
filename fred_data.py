import pandas_datareader.data as web
import pandas as pd
import datetime

# --- CONFIGURATION ---
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"


def get_macro_data():
    """
    Downloads Federal Funds Rate (Interest Rate) from FRED.
    Symbol: 'FEDFUNDS'
    """
    try:
        # 'FEDFUNDS' is the effective federal funds rate (monthly)
        df_macro = web.DataReader("FEDFUNDS", "fred", START_DATE, END_DATE)

        # Since macro data is monthly, we need to resample it to daily
        # to merge it with price data later. 'ffill' propagates the last value forward.
        df_macro = df_macro.resample('D').ffill()

        return df_macro
    except Exception as e:
        print(f"Error downloading FRED data: {e}")
        return pd.DataFrame()


# Example Usage
df_rates = get_macro_data()

# Let's say we have our Price Data (EURUSD)
# For demo purposes, creating a dummy dataframe
dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
df_prices = pd.DataFrame(index=dates, data={'Close': 1.1000})  # Dummy price

# Merge Price and Macro Data
# We align them by index (Date)
df_combined = df_prices.join(df_rates)
df_combined = df_combined.rename(columns={'FEDFUNDS': 'Interest_Rate'})

# Forward fill any missing weekend data
df_combined = df_combined.ffill()

print(df_combined.tail())

# --- LOGIC IMPLEMENTATION ---
# Example: Only trade LONG USD if Interest Rates are rising (Momentum)
current_rate = df_combined.iloc[-1]['Interest_Rate']
prev_rate = df_combined.iloc[-30]['Interest_Rate']  # Rate 30 days ago

if current_rate > prev_rate:
    print("Fundamental Bias: BULLISH USD (Rates are rising)")
    # set_strategy_bias("Short EURUSD")
else:
    print("Fundamental Bias: NEUTRAL/BEARISH USD")