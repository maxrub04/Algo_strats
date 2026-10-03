import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
#  EURO NIGHT STRATEGY  (ported from TradeStation EasyLanguage)
#  Book: Appendix B, pages 255-258
#
#  Bar size   : H1 (book uses 105-min; H1 is close enough)
#  Session    : 18:00 → 07:00 ET  (overnight window)
#  Last entry : 01:00 ET
#  Hard exit  : 07:00 ET  (before US open)
#
#  Entry logic:
#  - LongPrice  = MA(High, Nb) - ATRmult × ATR(NATR)  ← buy  limit below MA
#  - ShortPrice = MA(Low,  Nb) + ATRmult × ATR(NATR)  ← sell limit above MA
#  - Enter whichever is closest to current Close
#  - TP = Entry ± TRmult × TrueRange
#  - SL = Stoplo points
# ============================================================

# ─── PATH ────────────────────────────────────────────────────────────────────
# Point this at your actual file. The file has 7 tab-separated columns:
#   Time | Open | High | Low | Close | Volume | (spread/ticks col)
CSV_PATH = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/data/dukascopy/euro/EURUSD_H1.csv"

SYMBOL    = "EURUSD"
TIMEFRAME = "H1"

INITIAL_CAPITAL = 10_000.0
RISK_PERCENT    = 1.0       # % of balance risked per trade
POINT_VALUE     = 0.0001    # 1 pip for EURUSD

# Strategy params (book page 255 defaults)
Nb      = 10
NATR    = 60
ATRmult = 3.0
TRmult  = 0.5
Stoplo  = 275   # points  (275 × 0.0001 = 0.0275 = ~27.5 pips)

# Time windows (HHMM, 24h)
FIRST_TIME      = 1800   # 18:00 — session open
LAST_ENTRY_TIME = 100    # 01:00 — last new-entry bar
HARD_EXIT_TIME  = 700    # 07:00 — force-close all

# Date range
START_DATE = pd.to_datetime("2015-01-01")
END_DATE   = pd.to_datetime("2018-12-31")


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load Dukascopy H1 CSV (7 tab-separated columns, header row).
    Columns: Time | Open | High | Low | Close | Volume | Extra
    """
    df = pd.read_csv(
        csv_path, sep="\t", header=0,
        names=["DateTime", "Open", "High", "Low", "Close", "Volume", "Extra"],
        skiprows=1,          # skip the original header row
        parse_dates=["DateTime"],
        date_format="%Y-%m-%d %H:%M:%S",
    )
    df = df.drop(columns=["Extra"], errors="ignore")
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs(),
    ], axis=1).max(axis=1)


def prepare_indicators(df: pd.DataFrame,
                       nb: int, natr: int,
                       atr_mult: float) -> pd.DataFrame:
    df = df.copy()
    df["Time"] = df["DateTime"].dt.hour * 100 + df["DateTime"].dt.minute

    tr             = _true_range(df)
    df["TR"]       = tr
    df["ATR_band"] = tr.rolling(natr).mean()
    df["MA_High"]  = df["High"].rolling(nb).mean()
    df["MA_Low"]   = df["Low"].rolling(nb).mean()

    df["LongPrice"]  = df["MA_High"] - atr_mult * df["ATR_band"]
    df["ShortPrice"] = df["MA_Low"]  + atr_mult * df["ATR_band"]

    diff_long  = (df["Close"] - df["LongPrice"]).abs()
    diff_short = (df["Close"] - df["ShortPrice"]).abs()
    df["EntryToPick"] = np.where(diff_long <= diff_short, 1, 2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_euro_night_strategy(df_raw: pd.DataFrame,
                            nb:       int   = Nb,
                            natr:     int   = NATR,
                            atr_mult: float = ATRmult,
                            tr_mult:  float = TRmult,
                            stoplo:   int   = Stoplo) -> pd.DataFrame:
    df = prepare_indicators(df_raw, nb, natr, atr_mult)
    df = df.reset_index(drop=True)

    trades          = []
    current_balance = INITIAL_CAPITAL
    in_trade        = False
    position        = {}
    session_traded  = False   # max 1 trade per overnight session
    n               = len(df)

    for i in range(max(nb, natr) + 1, n):
        row       = df.iloc[i]
        bar_high  = row["High"]
        bar_low   = row["Low"]
        bar_close = row["Close"]
        bar_time  = row["Time"]
        bar_tr    = row["TR"]

        # Reset session flag at 18:00
        if bar_time == FIRST_TIME:
            session_traded = False

        # ── 1. MANAGE OPEN POSITION ──────────────────────────────────────
        if in_trade:
            d        = position["direction"]
            entry_px = position["entry_price"]
            stop_px  = position["stop_loss"]
            tp_px    = position["take_profit"]
            units    = position["units"]

            exit_price = None
            outcome    = ""

            if d == "Long":
                if bar_low <= stop_px:
                    exit_price, outcome = stop_px, "Loss"
                elif bar_high >= tp_px:
                    exit_price, outcome = tp_px, "Win"
            else:
                if bar_high >= stop_px:
                    exit_price, outcome = stop_px, "Loss"
                elif bar_low <= tp_px:
                    exit_price, outcome = tp_px, "Win"

            # Hard exit at 7 AM — before US session
            if outcome == "" and bar_time >= HARD_EXIT_TIME:
                exit_price = bar_close
                outcome    = "HardExit_7AM"

            if exit_price is not None:
                pnl = ((exit_price - entry_px) * units if d == "Long"
                       else (entry_px - exit_price) * units)
                current_balance += pnl
                trades.append({
                    "EntryDate"  : position["entry_date"],
                    "ExitDate"   : row["DateTime"],
                    "Direction"  : d,
                    "EntryPrice" : round(entry_px,   5),
                    "ExitPrice"  : round(exit_price, 5),
                    "StopLoss"   : round(stop_px,    5),
                    "TakeProfit" : round(tp_px,      5),
                    "Outcome"    : outcome,
                    "Profit"     : round(pnl, 2),
                    "Balance"    : round(current_balance, 2),
                })
                in_trade = False
                position = {}

        # ── 2. ENTRY LOGIC ───────────────────────────────────────────────
        # Night window crosses midnight: 18:00–00:59 OR 00:00–00:59
        in_entry_window = (bar_time >= FIRST_TIME or bar_time < LAST_ENTRY_TIME)

        if not in_trade and in_entry_window and not session_traded:
            pick        = row["EntryToPick"]
            long_price  = row["LongPrice"]
            short_price = row["ShortPrice"]

            filled    = False
            direction = ""
            entry_px  = 0.0

            if pick == 1 and bar_low <= long_price:
                entry_px, direction, filled = long_price, "Long", True
            elif pick == 2 and bar_high >= short_price:
                entry_px, direction, filled = short_price, "Short", True

            if filled:
                risk_dist = stoplo * POINT_VALUE
                if risk_dist <= 0:
                    continue

                if direction == "Long":
                    tp_px   = entry_px + tr_mult * bar_tr
                    stop_px = entry_px - risk_dist
                else:
                    tp_px   = entry_px - tr_mult * bar_tr
                    stop_px = entry_px + risk_dist

                # TP must be on profitable side
                if direction == "Long"  and tp_px <= entry_px:
                    continue
                if direction == "Short" and tp_px >= entry_px:
                    continue

                units = (current_balance * (RISK_PERCENT / 100.0)) / risk_dist

                in_trade       = True
                session_traded = True
                position = {
                    "entry_date" : row["DateTime"],
                    "direction"  : direction,
                    "entry_price": entry_px,
                    "stop_loss"  : stop_px,
                    "take_profit": tp_px,
                    "units"      : units,
                }

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
#  PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(results: pd.DataFrame, symbol: str):
    if results.empty:
        print("No trades generated.")
        return

    wins   = results[results["Profit"] > 0]
    losses = results[results["Profit"] < 0]
    total  = len(results)

    win_rate   = len(wins) / total * 100
    avg_win    = wins["Profit"].mean()   if len(wins)   > 0 else 0.0
    avg_loss   = losses["Profit"].mean() if len(losses) > 0 else 0.0
    expectancy = (avg_win * win_rate + avg_loss * (100 - win_rate)) / 100

    running_max = results["Balance"].cummax()
    dd          = (results["Balance"] - running_max) / running_max * 100
    max_dd      = dd.min()

    std    = results["Profit"].std()
    sharpe = (results["Profit"].mean() / std * np.sqrt(252)) if std > 0 else 0

    final   = results["Balance"].iloc[-1]
    ret_pct = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print("\n" + "=" * 60)
    print(f"  EURO NIGHT STRATEGY  |  {symbol}  |  {TIMEFRAME}")
    print("=" * 60)
    print(f"  Period        : {results['EntryDate'].min().date()} → {results['EntryDate'].max().date()}")
    print(f"  Total Trades  : {total}")
    print(f"  Win Rate      : {win_rate:.1f}%")
    print(f"  Avg Win       : ${avg_win:,.2f}")
    print(f"  Avg Loss      : ${avg_loss:,.2f}")
    print(f"  Expectancy    : ${expectancy:,.2f} per trade")
    print(f"  Final Balance : ${final:,.2f}")
    print(f"  Total Return  : {ret_pct:.2f}%")
    print(f"  Max Drawdown  : {max_dd:.2f}%")
    print(f"  Sharpe Ratio  : {sharpe:.3f}")
    print("=" * 60)
    print("\nOutcome breakdown:")
    print(results["Outcome"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results: pd.DataFrame, symbol: str):
    if results.empty:
        return

    sns.set(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Euro Night Strategy — {symbol} ({TIMEFRAME})",
                 fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(results["ExitDate"], results["Balance"], color="steelblue", lw=2)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Balance ($)")

    ax = axes[0, 1]
    running_max = results["Balance"].cummax()
    dd = (results["Balance"] - running_max) / running_max * 100
    ax.fill_between(results["ExitDate"], dd, 0, color="tomato", alpha=0.4)
    ax.set_title(f"Drawdown %  (Max: {dd.min():.2f}%)")
    ax.set_ylabel("Drawdown (%)")

    ax = axes[1, 0]
    sns.histplot(results["Profit"], kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Profit Distribution")

    ax = axes[1, 1]
    w = results[results["Profit"] > 0]["Profit"].cumsum().reset_index(drop=True)
    l = results[results["Profit"] < 0]["Profit"].cumsum().reset_index(drop=True)
    ax.plot(w, color="green", label="Cumulative Wins")
    ax.plot(l, color="red",   label="Cumulative Losses")
    ax.legend()
    ax.set_title("Win / Loss Separation")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETER SWEEP  (optional)
# ─────────────────────────────────────────────────────────────────────────────
def parameter_sweep(df_raw: pd.DataFrame):
    print("\n--- Parameter Sweep (ATRmult × TRmult) ---")
    grid = []
    for atm in [2.0, 2.5, 3.0, 3.5]:
        for trm in [0.3, 0.5, 0.7, 1.0]:
            r = run_euro_night_strategy(df_raw, atr_mult=atm, tr_mult=trm)
            if not r.empty:
                ret = (r["Balance"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                wr  = len(r[r["Profit"] > 0]) / len(r) * 100
                grid.append({"ATRmult": atm, "TRmult": trm,
                              "Return%": round(ret, 2),
                              "WinRate%": round(wr, 2),
                              "Trades": len(r)})

    df_grid = pd.DataFrame(grid)
    print(df_grid.to_string(index=False))

    pivot = df_grid.pivot(index="TRmult", columns="ATRmult", values="Return%")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
    plt.title("Return % — ATRmult × TRmult")
    plt.tight_layout()
    plt.show()
    return df_grid


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading {SYMBOL} {TIMEFRAME}...")

    df_ohlc = load_data(CSV_PATH)
    print(f"Total bars in file : {len(df_ohlc)}")
    print(f"Full range         : {df_ohlc['DateTime'].iloc[0].date()} → {df_ohlc['DateTime'].iloc[-1].date()}")

    # Filter date range
    df_ohlc = df_ohlc[
        (df_ohlc["DateTime"] >= START_DATE) &
        (df_ohlc["DateTime"] <= END_DATE)
    ].reset_index(drop=True)

    print(f"Bars after filter  : {len(df_ohlc)}")
    print(f"Filtered range     : {df_ohlc['DateTime'].iloc[0].date()} → {df_ohlc['DateTime'].iloc[-1].date()}")

    # 1. Run backtest
    results = run_euro_night_strategy(df_ohlc)

    # 2. Report
    print_report(results, SYMBOL)

    # 3. Save trades CSV
    out_dir  = "/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results"
    out_path = os.path.join(out_dir, f"{SYMBOL}_EuroNight_{TIMEFRAME}_trades.csv")
    os.makedirs(out_dir, exist_ok=True)
    if not results.empty:
        results.to_csv(out_path, index=False)
        print(f"\nTrades saved → {out_path}")

    # 4. Charts
    plot_results(results, SYMBOL)

    # 5. Uncomment for parameter sweep:
    # parameter_sweep(df_ohlc)