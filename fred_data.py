import pandas as pd
from fredapi import Fred
import numpy as np

# --- CONFIGURATION ---
API_KEY = 'YOUR_API'

INDICATORS_MAP = {
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
    'NAPM': 'ISM_PMI_Manufacturing'
}


# --- 1. DATA FETCHING ---
def fetch_fred_data(api_key, indicators):
    print("Connecting to FRED API...")
    try:
        fred = Fred(api_key=api_key)
    except Exception as e:
        print(f"Auth Error: {e}")
        return pd.DataFrame()

    data_frames = []
    for ticker, name in indicators.items():
        try:
            print(f"Fetching {ticker} -> {name}...")
            series = fred.get_series(ticker)
            series.name = name
            data_frames.append(series)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if not data_frames:
        return pd.DataFrame()

    print("Aligning data...")
    df = pd.concat(data_frames, axis=1)
    df = df.sort_index()
    # Forward fill is crucial because data comes at different frequencies (Daily vs Monthly)
    df_filled = df.ffill()
    # Filter from year 2000
    df_filled = df_filled[df_filled.index >= '2000-01-01']

    return df_filled


# --- 2. STRATEGY LOGIC ENGINE ---
def analyze_market_conditions(current, prev):
    """
    Applies the user's logic rules to a single row of data compared to previous.
    Returns a dictionary of signals and a text summary.
    """
    signals = []
    summary_text = []

    # Helper to safely get value (handles missing columns if download failed)
    def get_val(row, col):
        return row.get(col, 0)

    # --- RULE 1: NFP (Jobs) ---
    if get_val(current, 'NFP_Total_Jobs') > get_val(prev, 'NFP_Total_Jobs'):
        signals.append("NFP_UP")
        summary_text.append("[NFP] Jobs trend UP -> Bullish USD.")
    else:
        signals.append("NFP_DOWN")
        summary_text.append("[NFP] Jobs trend DOWN -> Bearish USD.")

    # --- RULE 2: US Treasury 2Y Yield ---
    if get_val(current, 'US_Treasury_2Y_Yield') > get_val(prev, 'US_Treasury_2Y_Yield'):
        signals.append("YIELD_2Y_UP")
        # Note: Usually rising nominal yields are Bearish Gold, but your logic said:
        summary_text.append("[2Y Yield] Rising -> Bearish Gold.")
    else:
        signals.append("YIELD_2Y_DOWN")
        summary_text.append("[2Y Yield] Falling -> Bullish Gold.")

    # --- RULE 3: VIX ---
    if get_val(current, 'VIX') > get_val(prev, 'VIX'):
        signals.append("VIX_RISK_OFF")
        summary_text.append("[VIX] Rising (Fear) -> Bullish Gold / Bearish Stocks.")
    else:
        signals.append("VIX_RISK_ON")
        summary_text.append("[VIX] Falling (Calm) -> Bearish Gold / Bullish Stocks.")

    # --- RULE 4: Trade Balance ---
    if get_val(current, 'Trade_Balance') > get_val(prev, 'Trade_Balance'):
        summary_text.append("[Trade] Balance Improving -> Bullish USD / Bullish Oil.")
    else:
        summary_text.append("[Trade] Balance Worsening -> Bearish USD / Bearish Oil.")

    # --- RULE 5: Inflation/Deflation (Exp/Imp) ---
    if (get_val(current, "Exports") > get_val(prev, "Exports")) and \
            (get_val(current, "Imports") < get_val(prev, "Imports")):
        signals.append("DEFLATION_PRESSURE")
        summary_text.append("[Macro] Deflationary signal (Exp UP / Imp DOWN).")
    else:
        summary_text.append("[Macro] Inflationary context.")

    # --- RULE 6: Jobless Claims (Detailed) ---
    init_up = get_val(current, 'Jobless_Claims_Initial') > get_val(prev, 'Jobless_Claims_Initial')
    cont_down = get_val(current, 'Jobless_Claims_Continuing') < get_val(prev, 'Jobless_Claims_Continuing')

    if init_up and cont_down:
        signals.append("CLAIMS_MIXED")
        summary_text.append("[CLAIMS] MIXED: Initial UP, Continuing DOWN -> Wait for Reversal.")
    elif init_up and not cont_down:
        signals.append("CLAIMS_BAD")
        summary_text.append("[CLAIMS] Economy Weakening -> Bullish Gold.")
    elif not init_up and cont_down:
        signals.append("CLAIMS_GOOD")
        summary_text.append("[CLAIMS] Economy Strengthening -> Bearish Gold.")
    else:
        summary_text.append("[CLAIMS] Neutral/Mixed.")

    # --- RULE 7: TIPS (Real Yields) ---
    # Most direct correlation to Gold
    tips_curr = get_val(current, 'TIPS_10Y_Real_Yield')
    tips_prev = get_val(prev, 'TIPS_10Y_Real_Yield')

    if tips_curr > tips_prev:
        signals.append("REAL_YIELD_UP")
        summary_text.append(f"[TIPS] Real Yield RISING ({tips_curr:.2f}%) -> STRONG SELL GOLD.")
    elif tips_curr < tips_prev:
        signals.append("REAL_YIELD_DOWN")
        summary_text.append(f"[TIPS] Real Yield FALLING ({tips_curr:.2f}%) -> STRONG BUY GOLD.")

    # --- RULE 8: PMI (ISM) ---
    pmi_curr = get_val(current, 'ISM_PMI_Manufacturing')
    pmi_prev = get_val(prev, 'ISM_PMI_Manufacturing')

    # Check if PMI column exists and is not 0/NaN
    if pmi_curr > 0:
        trend = "ACCELERATING" if pmi_curr > pmi_prev else "SLOWING"
        zone = "EXPANSION" if pmi_curr > 50 else "CONTRACTION"

        summary_text.append(f"[PMI] {pmi_curr} ({zone}). Trend: {trend}.")

        if pmi_curr > 50:
            summary_text.append("  -> Bullish USD / Bearish Gold.")
        else:
            summary_text.append("  -> Bearish USD / Bullish Gold.")

    return signals, "\n".join(summary_text)


# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":

    # A. Get Data
    df = fetch_fred_data(API_KEY, INDICATORS_MAP)

    if not df.empty:
        print("\n" + "=" * 40)
        print("STARTING MACRO ANALYSIS")
        print("=" * 40)

        # B. Apply Logic to History (Optional: creates a log for every day)
        # We will create a list to store results
        analysis_results = []

        # Iterate from the 2nd row to compare with previous
        # (This can be slow for 20 years, for speed use vectorization,
        # but for complex logic loops are clearer)
        for i in range(1, len(df)):
            current_date = df.index[i]
            current_row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # Check if there is ANY change in data (since we use ffill, many days are identical)
            # We only want to analyze when data updates
            if not current_row.equals(prev_row):
                signals_list, report = analyze_market_conditions(current_row, prev_row)
                analysis_results.append({
                    "Date": current_date,
                    "Report": report,
                    "Signals": signals_list
                })

        # Save Analysis History
        df_analysis = pd.DataFrame(analysis_results)
        #df_analysis.to_csv("macro_analysis_log.csv", index=False)
        print(f"Historical analysis saved to macro_analysis_log.csv ({len(df_analysis)} updates found).")

        # C. SHOW LATEST REPORT (What to do NOW)
        print("\n" + "*" * 40)
        print(f"LATEST SIGNAL REPORT: {df.index[-1].date()}")
        print("*" * 40)

        # Get the very last row and the one before it
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        _, final_report = analyze_market_conditions(last_row, prev_row)
        print(final_report)
        print("*" * 40)
    else:
        print("Data download failed. Check API Key.")
