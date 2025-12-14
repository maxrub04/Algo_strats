import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Set style for a professional look
plt.style.use('ggplot')

# --- DATE RANGE ---
start_date = pd.to_datetime("2013-01-20")
end_date = pd.to_datetime("2016-04-20")


def plot_performance(trades_df, initial_capital=10000):
    """
    Plots the Equity Curve and Percentage Return similar to the screenshot.
    Expects trades_df to have: 'Exit_Time' and 'Profit_Amount' (in $) columns.
    """

    # --- 1. PREPARE DATA ---
    df = trades_df.copy()
    df.sort_values("Exit_Time", inplace=True)

    # Calculate Cumulative Equity
    df['Cumulative_Profit'] = df['Profit_Amount'].cumsum()
    df['Equity'] = initial_capital + df['Cumulative_Profit']

    # Calculate Percentage Return
    df['Return_Pct'] = (df['Equity'] - initial_capital) / initial_capital * 100

    # Calculate Drawdown (for statistics)
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
    max_drawdown = df['Drawdown'].min() * 100  # In percent

    # Calculate Stats for the Text Box
    total_return = (df['Equity'].iloc[-1] - initial_capital) / initial_capital * 100
    end_capital = df['Equity'].iloc[-1]
    total_trades = len(df)
    win_rate = len(df[df['Profit_Amount'] > 0]) / total_trades * 100

    # --- 2. PLOTTING SETUP ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- TOP CHART: EQUITY CURVE (CAPITAL) ---
    ax1.plot(df['Exit_Time'], df['Equity'], color='blue', linewidth=1.5, label='Equity')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Start Capital')

    # Styling Top Chart
    ax1.set_title('Equity Curve - Capital', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Capital ($)', fontsize=12)
    ax1.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

    # --- STATISTICS BOX (Top Left) ---
    stats_text = (
        f"Statistics:\n"
        f"Total Return: {total_return:.2f}%\n"
        f"Max Drawdown: {max_drawdown:.2f}%\n"
        f"Start Capital: ${initial_capital:,.0f}\n"
        f"End Capital: ${end_capital:,.0f}\n"
        f"Trades: {total_trades}\n"
        f"Win Rate: {win_rate:.2f}%"
    )

    # Add text box with beige background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # --- BOTTOM CHART: PERCENTAGE RETURN ---
    ax2.plot(df['Exit_Time'], df['Return_Pct'], color='green', linewidth=1.5)

    # Fill area under the curve
    ax2.fill_between(df['Exit_Time'], df['Return_Pct'], 0,
                     where=(df['Return_Pct'] >= -999),  # Fill everything
                     color='green', alpha=0.3, label='Cumulative Return %')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Styling Bottom Chart
    ax2.set_title('Equity Curve - Percentage Return', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

    # Add Legend to bottom chart
    ax2.legend(loc='upper left')

    # --- DATE FORMATTING ---
    # Rotates dates and formats them nicely
    plt.xticks(rotation=45)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Month-Day format

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    trades = pd.read_csv('/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/backtest_results/BRENTCMDUSD_4h_trades.csv')

    trades["Date"] = pd.to_datetime(trades["Date"])

    """trades = trades[(trades["OpenTime"] >= start_date) &
                              (trades["OpenTime"] <= end_date)].reset_index(drop=True)"""

    dates = trades["Date"]
    profits = trades["Profit"]



    visual_trades = pd.DataFrame({
        'Exit_Time': dates,
        'Profit_Amount': profits
    })

    # RUN THE PLOT
    plot_performance(visual_trades, initial_capital=10000)