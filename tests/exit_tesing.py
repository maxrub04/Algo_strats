import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- CONFIGURATION ---
FOLDER = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data"
SYMBOL = "BRENTCMDUSD"
TIMEFRAME = "4h"
START_DATE = pd.to_datetime("2018-01-20")
END_DATE = pd.to_datetime("2020-04-20")


# --- DATA LOADING (Reused) ---
def load_data(symbol, tf):
    filepath = os.path.join(FOLDER, f"{symbol}_{tf}.csv")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    df = pd.read_csv(filepath, sep="\t", header=None, names=["DateTime", "Open", "High", "Low", "Close", "Vol"])
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df[(df["DateTime"] >= START_DATE) & (df["DateTime"] <= END_DATE)].reset_index(drop=True)
    return df


def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=period).mean()
    return df


# --- 1. ENTRY GENERATORS ---

def get_inside_bar_entries(df):
    """
    Generates 'Similar-Approach' entries (Inside Bar Breakout)
    Returns a list of dicts: {'index': i, 'type': 'Buy'/'Sell', 'price': price, 'sl': sl_price}
    """
    entries = []
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Is_Inside"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

    for i in range(1, len(df) - 1):
        if df.loc[i - 1, "Is_Inside"]:
            ib_high = df.loc[i - 1, "High"]
            ib_low = df.loc[i - 1, "Low"]
            curr_high = df.loc[i, "High"]
            curr_low = df.loc[i, "Low"]

            # Breakout Logic
            if curr_high > ib_high:
                entries.append({'index': i, 'type': 'Buy', 'price': ib_high, 'sl': ib_low})
            elif curr_low < ib_low:
                entries.append({'index': i, 'type': 'Sell', 'price': ib_low, 'sl': ib_high})
    return entries


def get_random_entries(df, num_trades=100):
    """
    Generates 'Random' entries to test if the Exit has a standalone edge.
    """
    entries = []
    indices = sorted(random.sample(range(14, len(df) - 100), num_trades))  # Avoid start/end

    for i in indices:
        direction = random.choice(['Buy', 'Sell'])
        current_close = df.loc[i, 'Close']
        atr = df.loc[i, 'ATR']

        # Artificial SL for random entry (1 ATR away)
        if direction == 'Buy':
            sl = current_close - atr
            entry_px = current_close
        else:
            sl = current_close + atr
            entry_px = current_close

        entries.append({'index': i, 'type': direction, 'price': entry_px, 'sl': sl})
    return entries


# --- 2. EXIT LOGIC TESTER ---

class ExitTester:
    def __init__(self, df):
        self.df = df
        self.risk_per_trade = 100.0  # Fixed $100 risk per trade

    def run_backtest(self, entries, exit_type="Fixed_RR", param1=2.0, param2=None):
        """
        exit_type:
            - 'Fixed_RR': param1 = Reward:Risk Ratio
            - 'Trailing_ATR': param1 = ATR Multiplier for Trail
            - 'Time_Exit': param1 = Bars to hold
        """
        results = []
        equity = 0

        for entry in entries:
            start_idx = entry['index']
            direction = entry['type']
            entry_price = entry['price']
            stop_loss = entry['sl']

            risk_dist = abs(entry_price - stop_loss)
            if risk_dist == 0: continue

            # Position Sizing (Fixed Risk)
            units = self.risk_per_trade / risk_dist

            # --- DEFINE EXIT PARAMETERS ---
            take_profit = 0
            trailing_stop = 0

            if exit_type == "Fixed_RR":
                rr_ratio = param1
                if direction == 'Buy':
                    take_profit = entry_price + (risk_dist * rr_ratio)
                else:
                    take_profit = entry_price - (risk_dist * rr_ratio)

            elif exit_type == "Trailing_ATR":
                atr_mult = param1
                # Initial Trail
                curr_atr = self.df.loc[start_idx, 'ATR']
                if direction == 'Buy':
                    trailing_stop = entry_price - (curr_atr * atr_mult)
                else:
                    trailing_stop = entry_price + (curr_atr * atr_mult)

            outcome_pnl = 0
            exit_idx = len(self.df) - 1
            max_fav_excursion = 0  # MFE
            max_adv_excursion = 0  # MAE

            # --- SIMULATE TRADE ---
            for j in range(start_idx + 1, len(self.df)):
                bar_high = self.df.loc[j, 'High']
                bar_low = self.df.loc[j, 'Low']
                bar_close = self.df.loc[j, 'Close']
                curr_atr = self.df.loc[j, 'ATR']

                # Update MAE/MFE
                if direction == "Buy":
                    mfe = (bar_high - entry_price)
                    mae = (entry_price - bar_low)  # Drawdown is positive here for calc
                else:
                    mfe = (entry_price - bar_low)
                    mae = (bar_high - entry_price)

                max_fav_excursion = max(max_fav_excursion, mfe)
                max_adv_excursion = max(max_adv_excursion, mae)

                # 1. CHECK STOP LOSS (Always active)
                hit_sl = False
                if direction == 'Buy' and bar_low <= stop_loss:
                    hit_sl = True
                elif direction == 'Sell' and bar_high >= stop_loss:
                    hit_sl = True

                if hit_sl:
                    outcome_pnl = -self.risk_per_trade
                    exit_idx = j
                    break

                # 2. CHECK EXIT STRATEGY`
                hit_tp = False

                if exit_type == "Fixed_RR":
                    if direction == 'Buy' and bar_high >= take_profit:
                        hit_tp = True
                    elif direction == 'Sell' and bar_low <= take_profit:
                        hit_tp = True

                    if hit_tp:
                        outcome_pnl = self.risk_per_trade * param1
                        exit_idx = j
                        break

                elif exit_type == "Time_Exit":
                    bars_held = j - start_idx
                    if bars_held >= param1:
                        # Close at Close
                        diff = (bar_close - entry_price) if direction == 'Buy' else (entry_price - bar_close)
                        outcome_pnl = diff * units
                        exit_idx = j
                        break

                elif exit_type == "Trailing_ATR":
                    # Update Trail
                    if direction == 'Buy':
                        new_trail = bar_close - (curr_atr * param1)
                        if new_trail > trailing_stop: trailing_stop = new_trail
                        if bar_low <= trailing_stop:  # Hit Trail
                            diff = (trailing_stop - entry_price)
                            outcome_pnl = diff * units
                            exit_idx = j
                            break
                    else:
                        new_trail = bar_close + (curr_atr * param1)
                        if new_trail < trailing_stop: trailing_stop = new_trail
                        if bar_high >= trailing_stop:  # Hit Trail
                            diff = (entry_price - trailing_stop)
                            outcome_pnl = diff * units
                            exit_idx = j
                            break

            results.append({
                'Profit': outcome_pnl,
                'MFE': max_fav_excursion * units,
                'MAE': max_adv_excursion * units,  # in Dollars
                'Bars': exit_idx - start_idx
            })

        return pd.DataFrame(results)


# --- 3. ROBUSTNESS / CORE SYSTEM TESTING ---

def run_parameter_sweep(df, entries):
    """
    The 'Core System Testing' Matrix.
    Tests Reward:Risk ratios vs Breakout variations (simulated by SL size here for simplicity)
    """
    print("\n--- Running Core System Robustness Test (Heatmap) ---")

    rr_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    # We will vary the 'Risk' multiplier.
    # Current strategy uses IB High/Low. Let's assume we modify the stop width slightly
    # to simulate 'variable' breakout parameters or just test pure R:R robustness.

    matrix_res = []

    tester = ExitTester(df)

    for rr in rr_ratios:
        # Test Fixed R:R Exit
        res = tester.run_backtest(entries, exit_type="Fixed_RR", param1=rr)
        if not res.empty:
            total_profit = res['Profit'].sum()
            win_rate = len(res[res['Profit'] > 0]) / len(res)
            matrix_res.append({'RR': rr, 'Total_Profit': total_profit, 'Win_Rate': win_rate})

    return pd.DataFrame(matrix_res)


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    df = load_data(SYMBOL, TIMEFRAME)
    df = calculate_atr(df)

    if df.empty:
        print("No Data Found.")
    else:
        # 1. PREPARE ENTRIES
        print("Generating Entries...")
        real_entries = get_inside_bar_entries(df)
        random_entries = get_random_entries(df, num_trades=len(real_entries))  # Match count

        tester = ExitTester(df)

        # 2. RUN EXIT COMPARISON (Similar-Approach vs Random)
        print(f"\nComparing Exits on {len(real_entries)} trades...")

        scenarios = [
            ("Fixed RR 2.75", "Fixed_RR", 2.75),
            ("Fixed RR 3.0", "Fixed_RR", 3.0),
            ("Trailing ATR 3x", "Trailing_ATR", 3.0),
            ("Time Exit 8 Bars", "Time_Exit", 8)
        ]

        comparison_data = []

        for name, etype, param in scenarios:
            # Test on Real Entry
            res_real = tester.run_backtest(real_entries, etype, param)
            prof_real = res_real['Profit'].sum()
            eff_real = res_real['Profit'].sum() / res_real['MFE'].sum() if res_real['MFE'].sum() != 0 else 0

            # Test on Random Entry
            res_rand = tester.run_backtest(random_entries, etype, param)
            prof_rand = res_rand['Profit'].sum()

            comparison_data.append({
                "Strategy": name,
                "Real_Entry_Profit": prof_real,
                "Random_Entry_Profit": prof_rand,
                "Real_Entry_Efficiency": eff_real
            })

        df_comp = pd.DataFrame(comparison_data)

        # 3. VISUALIZE EXIT COMPARISON
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # Bar Chart: Profitability
        x = np.arange(len(df_comp))
        width = 0.35
        ax[0].bar(x - width / 2, df_comp['Real_Entry_Profit'], width, label='Real Entry', color='green')
        ax[0].bar(x + width / 2, df_comp['Random_Entry_Profit'], width, label='Random Entry', color='gray')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(df_comp['Strategy'])
        ax[0].set_title("Profitability: Real vs Random Entry")
        ax[0].set_ylabel("Net Profit ($)")
        ax[0].legend()
        ax[0].axhline(0, color='black', linewidth=0.8)

        # Scatter: MFE vs MAE (For the best Real Strategy)
        best_strat_idx = df_comp['Real_Entry_Profit'].idxmax()
        best_name = df_comp.loc[best_strat_idx, 'Strategy']
        # Re-run to get details
        s_type = scenarios[best_strat_idx][1]
        s_param = scenarios[best_strat_idx][2]
        res_best = tester.run_backtest(real_entries, s_type, s_param)

        ax[1].scatter(res_best['MAE'], res_best['MFE'], alpha=0.5, c=np.where(res_best['Profit'] > 0, 'green', 'red'))
        ax[1].set_title(f"MFE vs MAE ({best_name})")
        ax[1].set_xlabel("Max Adverse Excursion ($ Risk)")
        ax[1].set_ylabel("Max Favorable Excursion ($ Potential)")
        ax[1].plot([0, 200], [0, 200], ls="--", c=".3")  # 1:1 Line

        plt.tight_layout()
        plt.show()

        # 4. CORE SYSTEM TESTING (Parameter Sweep)
        # Testing RR 1.0 to 6.0
        robustness_df = run_parameter_sweep(df, real_entries)

        plt.figure(figsize=(10, 5))
        sns.barplot(x='RR', y='Total_Profit', data=robustness_df, palette="viridis")
        plt.axhline(0, color='red')
        plt.title("Core System Robustness: Profitability across Risk:Reward Ratios")
        plt.ylabel("Total Profit ($)")
        plt.xlabel("Risk : Reward Ratio")
        plt.show()

        print("\n--- ROBUSTNESS DATA ---")
        print(robustness_df)

        # Check if profitable in > 70% of cases (Simple check on RR sweep)
        profitable_runs = len(robustness_df[robustness_df['Total_Profit'] > 0])
        total_runs = len(robustness_df)
        print(f"\nRobustness Score: {profitable_runs}/{total_runs} configurations were profitable.")
        if profitable_runs / total_runs >= 0.7:
            print(">> SYSTEM PASSES CORE ROBUSTNESS TEST ( > 70% )")
        else:
            print(">> SYSTEM FAILS CORE ROBUSTNESS TEST ( < 70% )")