import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set style
sns.set(style="darkgrid")


def perform_advanced_eda(df_trades):
    if df_trades.empty:
        print("No trades to analyze.")
        return

    # --- PRE-PROCESSING ---
    df_trades['DateTime'] = pd.to_datetime(df_trades['Date'])
    df_trades['Hour'] = df_trades['DateTime'].dt.hour
    df_trades['DayOfWeek'] = df_trades['DateTime'].dt.day_name()
    # Normalize day order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)

    # 1. EQUITY CURVE (Top Wide)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_trades['DateTime'], df_trades['Balance'], color='green', linewidth=2)
    ax1.set_title(f"Equity Curve (Final Balance: ${df_trades['Balance'].iloc[-1]:,.2f})", fontsize=14)
    ax1.set_ylabel("Balance ($)")

    # 2. PROFIT DISTRIBUTION (Histogram)
    ax2 = fig.add_subplot(gs[1, 0])
    sns.histplot(df_trades['Profit'], kde=True, ax=ax2, color='blue')
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("Distribution of Trade Profits", fontsize=12)

    # 3. MACRO SCORE IMPACT (Box Plot) - CRITICAL FOR YOU
    # Does the Fundamental Score correlate with Profit?
    ax3 = fig.add_subplot(gs[1, 1])
    if 'Macro_Score' in df_trades.columns:
        sns.boxplot(x='Macro_Score', y='Profit', data=df_trades, ax=ax3, palette="coolwarm")
        ax3.set_title("Profit vs. Macro Score (Is the filter working?)", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No Macro_Score Data", ha='center')

    # 4. DRAWDOWN CHART (Underwater Plot)
    ax4 = fig.add_subplot(gs[1, 2])
    running_max = df_trades['Balance'].cummax()
    drawdown = (df_trades['Balance'] - running_max) / running_max * 100
    ax4.fill_between(df_trades['DateTime'], drawdown, 0, color='red', alpha=0.3)
    ax4.set_title(f"Drawdown % (Max: {drawdown.min():.2f}%)", fontsize=12)

    # 5. HEATMAP: DAY vs HOUR (When do we win?)
    ax5 = fig.add_subplot(gs[2, 0])
    # Pivot table for heatmap
    pivot = df_trades.pivot_table(index='DayOfWeek', columns='Hour', values='Profit', aggfunc='sum')
    # Reindex days
    pivot = pivot.reindex(days_order)
    sns.heatmap(pivot, cmap="RdYlGn", center=0, ax=ax5, annot=False)
    ax5.set_title("Profit Heatmap (Day vs Hour)", fontsize=12)

    # 6. CUMULATIVE WINS vs LOSSES
    ax6 = fig.add_subplot(gs[2, 1])
    wins = df_trades[df_trades['Profit'] > 0]['Profit'].cumsum().reset_index(drop=True)
    losses = df_trades[df_trades['Profit'] < 0]['Profit'].cumsum().reset_index(drop=True)
    ax6.plot(wins, color='green', label='Cumulative Wins')
    ax6.plot(losses, color='red', label='Cumulative Losses')
    ax6.legend()
    ax6.set_title("Win/Loss Separation", fontsize=12)

    # 7. ROLLING WIN RATE (Stability)
    ax7 = fig.add_subplot(gs[2, 2])
    # 20-trade rolling average of Win (1) vs Loss (0)
    df_trades['Win'] = np.where(df_trades['Profit'] > 0, 1, 0)
    df_trades['Rolling_WR'] = df_trades['Win'].rolling(window=20).mean()
    ax7.plot(df_trades['Ticket'], df_trades['Rolling_WR'], color='purple')
    ax7.axhline(0.5, color='gray', linestyle='--')
    ax7.set_ylim(0, 1)
    ax7.set_title("Rolling Win Rate (20 Trades)", fontsize=12)

    plt.tight_layout()
    plt.show()

# --- HOW TO RUN IT ---
# Assuming you just ran your strategy and have 'df_trades'
# perform_advanced_eda(results)