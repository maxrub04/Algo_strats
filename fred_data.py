import pandas as pd
from fredapi import Fred

# --- CONFIGURATION ---
# Replace with your actual 32-character API key
API_KEY = 'api'

# Your requested mapping
INDICATORS_MAP = {
    'FEDFUNDS': 'Interest_Rate',  # Federal Funds Rate (Monthly)
    'CPIAUCSL': 'CPI',  # Consumer Price Index (Monthly)
    'PPIACO': 'PPI_Commodities',  # Producer Price Index (Monthly)
    'UNRATE': 'Unemployment_Rate',  # Unemployment Rate (Monthly)
    'PAYEMS': 'NFP_Total_Jobs',  # Non-Farm Payrolls (Monthly)
    'ICSA': 'Jobless_Claims_Initial',  # Initial Claims (Weekly)
    'CCSA': 'Jobless_Claims_Continuing',  # Continuing Claims (Weekly)
    'BOPGSTB': 'Trade_Balance',  # Trade Balance (Monthly)
    'EXPGS': 'Exports',  # Exports (Monthly)
    'IMPGS': 'Imports',  # Imports (Monthly)
    'DGS2': 'US_Treasury_2Y_Yield',  # 2Y Yield (Daily)
    'VIXCLS': 'VIX',  # VIX (Daily Close)
    'DFII10': 'TIPS_10Y_Real_Yield',  # Real Yield (Daily)
    'NAPM': 'ISM_PMI_Manufacturing'  # ISM PMI (Monthly)
}


def fetch_fred_data(api_key, indicators):
    print("Connecting to FRED API...")
    fred = Fred(api_key=api_key)

    data_frames = []

    for ticker, name in indicators.items():
        try:
            print(f"Fetching {ticker} -> {name}...")
            # get_series returns a pandas Series
            series = fred.get_series(ticker)
            series.name = name
            data_frames.append(series)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            print("Note: Some series like NAPM might be discontinued or restricted.")

    if not data_frames:
        return pd.DataFrame()

    print("Aligning data...")
    # 1. Concatenate all series into one DataFrame (Outer Join)
    df = pd.concat(data_frames, axis=1)

    # 2. Sort by Date
    df = df.sort_index()

    # 3. Handle Frequency Mismatch (Normalization)
    # Since we have Daily, Weekly, and Monthly data mixed:
    # We usually Forward Fill (ffill) so that on any given day,
    # the system sees the "last known" economic value.
    df_filled = df.ffill()

    # Optional: Drop rows before 2000 if you want to save memory
    df_filled = df_filled[df_filled.index >= '2000-01-01']

    return df_filled


# --- MAIN EXECUTION ---
if __name__ == "__main__":
        df_macro = fetch_fred_data(API_KEY, INDICATORS_MAP)

        if not df_macro.empty:
            print("-" * 30)
            print("Data Fetch Complete.")
            print(df_macro.tail())  # Show last 5 rows

            # Save to CSV for your backtest
            df_macro.to_csv("macro_data.csv")
            print("Saved to macro_data.csv")

#rules
"""if current_data['NFP_Total_Jobs'] > prev_data['NFP_Total_Jobs']:
    print(f"[NFP] Jobs trend is UP. Bullish USD.")
else:
    print(f"[NFP] Jobs trend is DOWN. Bearish USD.")

if current_data['US_Treasury_2Y_Yield'] > prev_data['US_Treasury_2Y_Yield']:
    print(f"[TIPS] Real Yields rising. Bearish Gold.")
else:
    print(f"[TIPS] Real Yields falling. Bullish Gold.")

if current_data["VIX"] > prev_data["VIX"]:
    print(f"[VIX] VIX rising. ?Bearish USD?. Bullish Gold")
else:
    print(f"[VIX] VIX falling. ?Bullish USD?. Bearish Gold")

if current_data["Trade_Balance"] > prev_data["Trade_Balance"]:
    print(f"[Trade Balance] Trade Balance raising. Bullish USD. Bullish Oil")
else:
    print(f"[Trade Balance] Trade Balance falling. Bearish USD. Bearish Oil")

if (current_data["Exports"] > prev_data["Exports"]) and (current_data["Imports"] < prev_data["Imports"]):
    print(f"Deflation.")
else:
    print(f"Inflation.")

# 1. LOGIC: JOBLESS CLAIMS (Specific Request)
# Initial Claims UP (Bad), Continuing Claims DOWN (Good) -> Mixed Signal -> Reversal
initial_up = current_data['Jobless_Claims_Initial'] > prev_data['Jobless_Claims_Initial']
continuing_down = current_data['Jobless_Claims_Continuing'] < prev_data['Jobless_Claims_Continuing']

if initial_up and continuing_down:
    print(f"[CLAIMS] MIXED SIGNAL: Initial UP, Continuing DOWN.")
    print(f"ACTION: Wait 1 hour -> Expect REVERSAL for Gold.")
elif initial_up and not continuing_down:
    print(f"[CLAIMS] Both metrics Bad for Economy. Clear Bullish Gold.")
elif not initial_up and continuing_down:
    print(f"[CLAIMS] Both metrics Good for Economy. Clear Bearish Gold.")
else:
    print(f"[CLAIMS] Neutral/Mixed (Initial Down, Continuing Up).")

if current_data['TIPS_10Y_Real_Yield'] > prev_data['TIPS_10Y_Real_Yield']:
    print(f"[TIPS] Real Yields are RISING ({current_data['TIPS_10Y_Real_Yield']:.2f}%).")
    print(f"Action: BEARISH GOLD (Sell XAUUSD). Opportunity cost of holding Gold is increasing.")
elif current_data['TIPS_10Y_Real_Yield'] < prev_data['TIPS_10Y_Real_Yield']:
    print(f"[TIPS] Real Yields are FALLING ({current_data['TIPS_10Y_Real_Yield']:.2f}%).")
    print(f"Action: BULLISH GOLD (Buy XAUUSD). Gold becomes attractive.")
else:
    print(f"[TIPS] Real Yields unchanged.")


current_pmi = current_data['ISM_PMI_Manufacturing']
prev_pmi    = prev_data['ISM_PMI_Manufacturing']

# 1. Базовое правило 50
if current_pmi > 50:
    print(f"[PMI] {current_pmi} > 50. Economy is EXPANDING.")
    print("  -> Bullish USD (Strong economy).")
    print("  -> Bearish Gold (Risk On).")
else:
    print(f"[PMI] {current_pmi} < 50. Economy is CONTRACTING.")
    print("  -> Bearish USD (Recession risk).")
    print("  -> Bullish Gold (Safe Haven demand).")

# 2. Правило Импульса (Momentum)
# Даже если PMI = 55 (рост), но месяц назад был 60 -> это замедление.
if current_pmi > prev_pmi:
    print(f"  -> [Trend] Activity is ACCELERATING (Better than last month).")
else:
    print(f"  -> [Trend] Activity is SLOWING DOWN (Worse than last month).")"""
