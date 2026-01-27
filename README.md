# Algo_strats: FX Algorithmic Trading

## Description
**Algo_strats** is a repository dedicated to the research, development, and rigorous backtesting of algorithmic trading strategies for the **Foreign Exchange (FX) market**.

The project aims to combine traditional financial analysis with modern computational techniques to identify profitable trading opportunities while prioritizing statistical significance and capital preservation.

## Strategy Components
The repository explores three core pillars of strategy generation:

* **Fundamental Analysis **
    * Analysis of macroeconomic indicators and news sentiment.
    * Currency correlation studies.
* **Technical Analysis **
    * Standard indicators (RSI, MACD, Bollinger Bands).
    * Price action and trend-following algorithms.

## Validation & Robustness
To prevent overfitting and ensure strategies perform well in live markets, rigorous testing protocols are applied:

### Out-of-Sample Testing
* Strict separation of data into **Training** and **Testing** sets.
* Strategies are optimized on training data and verified on unseen (out-of-sample) data to confirm true predictive power.

### Walk-Forward Analysis
* Implementation of **Rolling Window** validation.
* Strategies are re-optimized periodically on a sliding window of historical data to adapt to changing market regimes and assess parameter stability over time.

### Monte Carlo Simulations
* Resampling of trade sequences to assess the role of luck vs. skill.
* Generating thousands of alternative equity curves to estimate the probability of **Max Drawdown** and **Risk of Ruin** under different market conditions.

## Performance Metrics
Strategies are evaluated based on strict risk-adjusted return metrics, not just total profit:

* **Sharpe Ratio:** Used to evaluate the return of the strategy relative to its volatility (risk). We aim for a Sharpe Ratio > 1.5 to ensure consistent returns per unit of risk.
* **Calmar Ratio:** Measures the strategy's compound annual growth rate (CAGR) relative to its Maximum Drawdown. This is crucial for assessing the strategy's recovery ability after losses.
* **Profit Factor:** The ratio of gross profit to gross loss.
* **Win/Loss Ratio:** Analyzing the frequency of winning trades versus losing trades.

## Risk Management
Capital preservation is the foundation of every strategy in this repository. Key mechanisms include:

* **Stop-Loss Protocols:** Hard stops and trailing stops to limit downside exposure per trade.
* **Drawdown Control:** Circuit breakers that halt trading if equity drops below a predefined threshold.
