import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CSV_PATH = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results/BRENTCMDUSD_4h_trades.csv"  # Ensure this path is correct
STARTING_EQUITY = 10000.0
NUM_SIMULATIONS = 2000
TRADES_PER_YEAR = 100
RUIN_THRESHOLD = 6000.0


def run_monte_carlo(csv_path, start_equity, num_sims, horizon_trades):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
        if "Profit" not in df.columns:
            print("Error: CSV must have a 'Profit' column.")
            return

        pnl_pool = df["Profit"].values

        if len(pnl_pool) < 10:
            print("Not enough trades in CSV to run a reliable simulation.")
            return

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(pnl_pool)} historical trades. Running {num_sims} simulations...")

    # 2. Storage for results
    final_equities = []
    max_drawdowns = []
    returns_pct = []
    ret_dd_ratios = []
    is_ruined_count = 0

    all_curves = []

    # 3. Simulation Loop
    for i in range(num_sims):
        # A. Random Sample
        daily_pnl = np.random.choice(pnl_pool, size=horizon_trades, replace=True)

        # B. Construct Equity Curve
        equity_curve = np.zeros(horizon_trades + 1)
        equity_curve[0] = start_equity
        equity_curve[1:] = start_equity + np.cumsum(daily_pnl)

        # C. Check for Ruin
        if np.min(equity_curve) < RUIN_THRESHOLD:
            is_ruined_count += 1

        # D. Metrics
        final_eq = equity_curve[-1]
        total_profit = final_eq - start_equity
        ret_pct = (total_profit / start_equity) * 100

        # Max Drawdown %
        running_max = np.maximum.accumulate(equity_curve)
        dd_curve = (equity_curve - running_max) / running_max * 100
        max_dd = abs(np.min(dd_curve))

        # Return / Drawdown Ratio Calculation
        if max_dd == 0:
            rd_ratio = 0  # Avoid division by zero
        else:
            rd_ratio = ret_pct / max_dd

        final_equities.append(final_eq)
        max_drawdowns.append(max_dd)
        returns_pct.append(ret_pct)
        ret_dd_ratios.append(rd_ratio)  # <--- NEW: Store it

        if i < 100:
            all_curves.append(equity_curve)

    # 4. Generate Statistics
    results = {
        "Metric": ["Worst Case (95%)", "Below Avg (75%)", "Median (50%)", "Above Avg (25%)", "Best Case (5%)"],

        "Final Equity ($)": [
            np.percentile(final_equities, 5),
            np.percentile(final_equities, 25),
            np.percentile(final_equities, 50),
            np.percentile(final_equities, 75),
            np.percentile(final_equities, 95)
        ],

        "Return (%)": [
            np.percentile(returns_pct, 5),
            np.percentile(returns_pct, 25),
            np.percentile(returns_pct, 50),
            np.percentile(returns_pct, 75),
            np.percentile(returns_pct, 95)
        ],

        "Max Drawdown (%)": [
            np.percentile(max_drawdowns, 95),  # 95th percentile is the WORST (Deepest) DD
            np.percentile(max_drawdowns, 75),
            np.percentile(max_drawdowns, 50),
            np.percentile(max_drawdowns, 25),
            np.percentile(max_drawdowns, 5)
        ],

        "Return/DD Ratio": [
            np.percentile(ret_dd_ratios, 5),
            np.percentile(ret_dd_ratios, 25),
            np.percentile(ret_dd_ratios, 50),
            np.percentile(ret_dd_ratios, 75),
            np.percentile(ret_dd_ratios, 95)
        ]
    }

    df_res = pd.DataFrame(results)

    prob_profit = np.sum(np.array(returns_pct) > 0) / num_sims * 100
    prob_ruin = (is_ruined_count / num_sims) * 100

    # 5. Print Report
    print("\n" + "=" * 60)
    print(f"MONTE CARLO SIMULATION RESULTS ({num_sims} Runs)")
    print(f"Input: {len(pnl_pool)} trades | Horizon: {horizon_trades} trades")
    print("=" * 60)
    print(f"Start Equity:      ${start_equity:,.2f}")
    print(f"Ruin Threshold:    ${RUIN_THRESHOLD:,.2f}")
    print(f"Prob of Ruin:      {prob_ruin:.2f}%")
    print(f"Prob of Profit:    {prob_profit:.2f}%")
    print("-" * 60)

    pd.options.display.float_format = '{:,.2f}'.format
    print(df_res.to_string(index=False))
    print("-" * 60)




# --- EXECUTION ---
if __name__ == "__main__":
    target_csv = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results/BRENTCMDUSD_4h_trades.csv"  # Update your file name here
    run_monte_carlo(target_csv, STARTING_EQUITY, NUM_SIMULATIONS, TRADES_PER_YEAR)