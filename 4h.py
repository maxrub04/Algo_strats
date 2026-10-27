"""
MetaTrader 5 Live Trading Bot
Inside Bar стратегия + макро-фильтр (FRED)

FIX: Проверка маржи перед открытием + верификация фактического объёма
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import sys
import io
import os
import ssl
import urllib.request

from typing import Optional, Dict, Tuple, List
from fredapi import Fred

# ---------------------------------------------------------------------------
# FIX: UTF-8 stdout/stderr на Windows
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


# =============================================================================
# CONFIGURATION
# =============================================================================

class TradingConfig:
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "[password]")
    MT5_SERVER = os.getenv("MT5_SERVER", "OANDATMS-MT5")

    SYMBOL = os.getenv("SYMBOL", "OILBRNT.pro")
    TIMEFRAME = mt5.TIMEFRAME_H4
    MAGIC_NUMBER = 1234

    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "50000"))  # PLN
    RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.1"))
    RISK_REWARD = float(os.getenv("RISK_REWARD", "2.75"))
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))

    ACCOUNT_CURRENCY = ""

    FRED_API_KEY = os.getenv("FRED_API_KEY", "[api]")
    USE_MACRO_FILTER = os.getenv("USE_MACRO_FILTER", "true").lower() == "true"

    TRADING_START_HOUR = 2
    TRADING_END_HOUR = 23

    CHECK_INTERVAL = 1
    STATUS_LOG_INTERVAL = 600
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "trading_bot.log")

    ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "[api]")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "[chat_id]")


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = getattr(logging, TradingConfig.LOG_LEVEL, logging.INFO)

    logger = logging.getLogger("MT5_TradingBot")
    logger.setLevel(level)
    logger.handlers.clear()

    fh = logging.FileHandler(TradingConfig.LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)

    return logger


logger = setup_logging()


def setup_trade_logger():
    trade_logger = logging.getLogger("TRADES")
    trade_logger.setLevel(logging.INFO)
    handler = logging.FileHandler("trades.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    trade_logger.addHandler(handler)
    return trade_logger


trade_logger = setup_trade_logger()


# =============================================================================
# SSL патч
# =============================================================================

def _patch_ssl_globally():
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)
    logger.info("SSL opener patched OK")


_patch_ssl_globally()


# =============================================================================
# MT5 CONNECTION
# =============================================================================

class MT5Connection:
    @staticmethod
    def initialize() -> bool:
        if not mt5.initialize():
            logger.error("MT5 initialize failed: %s", mt5.last_error())
            return False
        logger.info("MT5 version: %s", mt5.version())
        return True

    @staticmethod
    def login() -> bool:
        ok = mt5.login(
            login=TradingConfig.MT5_LOGIN,
            password=TradingConfig.MT5_PASSWORD,
            server=TradingConfig.MT5_SERVER,
        )
        if not ok:
            logger.error("MT5 login failed: %s", mt5.last_error())
            return False
        acc = mt5.account_info()
        TradingConfig.ACCOUNT_CURRENCY = acc.currency
        logger.info("Logged in: account #%s | Balance %.2f %s | Equity %.2f %s",
                    acc.login, acc.balance, acc.currency, acc.equity, acc.currency)
        return True

    @staticmethod
    def shutdown():
        mt5.shutdown()
        logger.info("MT5 connection closed")


# =============================================================================
# MACRO
# =============================================================================
class MacroDataProcessor:
    """Processes macroeconomic data"""

    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key)
        self.indicators = {
            'FEDFUNDS': 'Interest_Rate',
            'CPIAUCSL': 'CPI',
            'PAYEMS': 'NFP_Total_Jobs',
            'DFII10': 'TIPS_10Y_Real_Yield',
            'EXPGS': 'Exports',
            'IMPGS': 'Imports',
            'BOPGSTB': 'Trade_Balance',
        }

    def fetch_and_process(self) -> pd.DataFrame:
        """Fetch macro data and calculate sentiment score"""
        print("Fetching Macro Data...")
        data_frames = []

        for ticker, name in self.indicators.items():
            try:
                series = self.fred.get_series(ticker)
                series.name = name
                data_frames.append(series)
            except Exception as e:
                print(f"Skipping {ticker}: {e}")

        if not data_frames:
            return pd.DataFrame()

        # Align and fill
        df = pd.concat(data_frames, axis=1).sort_index().ffill().dropna()

        # Calculate USD Score
        df['USD_Score'] = self._calculate_usd_score(df)

        df.index.name = 'DateTime'
        return df.reset_index()[['DateTime', 'USD_Score']]

    def _calculate_usd_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate USD sentiment score from economic indicators"""
        score = pd.Series(0, index=df.index)

        # NFP Trend
        if 'NFP_Total_Jobs' in df.columns:
            nfp_change = df['NFP_Total_Jobs'].diff()
            score += np.where(nfp_change > 0, 1, 0)
            score += np.where(nfp_change < 0, -1, 0)

        # TIPS Real Yield
        if 'TIPS_10Y_Real_Yield' in df.columns:
            tips_change = df['TIPS_10Y_Real_Yield'].diff()
            score += np.where(tips_change > 0, 1, 0)
            score += np.where(tips_change < 0, -1, 0)

        # Trade Balance
        if 'Trade_Balance' in df.columns:
            trade_change = df['Trade_Balance'].diff()
            score += np.where(trade_change > 0, 1, 0)
            score += np.where(trade_change < 0, -1, 0)

        # Import/Export dynamics
        if 'Imports' in df.columns and 'Exports' in df.columns:
            imp_change = df['Imports'].diff()
            exp_change = df['Exports'].diff()
            # Deflation scenario (good for USD)
            score += np.where((imp_change < 0) & (exp_change > 0), 1, 0)
            # Inflation scenario (bad for USD)
            score += np.where((imp_change > 0) & (exp_change < 0), -1, 0)

        return score

    @staticmethod
    def merge_with_ohlc(df_ohlc: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
        """Merge macro data with OHLC data"""
        df_ohlc = df_ohlc.sort_values("DateTime")
        df_macro = df_macro.sort_values("DateTime")

        df_merged = pd.merge_asof(df_ohlc, df_macro, on="DateTime", direction="backward")
        df_merged['USD_Score'] = df_merged['USD_Score'].fillna(0)

        return df_merged

# =============================================================================
# TECHNICALS
# =============================================================================

class TechnicalIndicators:

    @staticmethod
    def identify_inside_bar(df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """
         Inside Bar:
          df.iloc[-3] = Mother Bar
          df.iloc[-2] = Inside Bar  (high < mother.high AND low > mother.low)
          df.iloc[-1] = Current Bar (breakout candle)
        """
        if len(df) < 3:
            return False, None

        mother_bar = df.iloc[-3]
        inside_bar = df.iloc[-2]
        current_bar = df.iloc[-1]

        is_inside = (
                inside_bar["high"] <= mother_bar["high"] and
                inside_bar["low"] >= mother_bar["low"]
        )
        if not is_inside:
            return False, None

        ib_high = inside_bar["high"]
        ib_low = inside_bar["low"]
        signal = {}

        if current_bar["high"] > ib_high + 0.01:
            logger.debug(f" VALID BUY breakout: curr_H={current_bar['high']:.5f} > IB_H={ib_high:.5f}")
            signal = {
                "direction": "buy",
                "entry_price": ib_high,
                "stop_loss": ib_low,
                "inside_bar_time": inside_bar.name,
            }
        elif current_bar["low"] < ib_low - 0.01:
            logger.debug(f"VALID SELL breakout: curr_L={current_bar['low']:.5f} < IB_L={ib_low:.5f}")
            signal = {
                "direction": "sell",
                "entry_price": ib_low,
                "stop_loss": ib_high,
                "inside_bar_time": inside_bar.name,
            }

        if not signal:
            return False, None

        return True, signal


# =============================================================================
# TRADER
# =============================================================================

class InsideBarTrader:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.macro = MacroDataProcessor(config.FRED_API_KEY) if config.USE_MACRO_FILTER else None
        self._last_signal_bar_time: Optional[pd.Timestamp] = None
        self._last_signal_ib_time: Optional[pd.Timestamp] = None
        self._max_entry_deviation = 0.0003

    # ── market data ──────────────────────────────────────────────────────────
    def _get_bars(self, count: int = 100) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(self.config.SYMBOL, self.config.TIMEFRAME, 0, count)
        if rates is None or len(rates) == 0:
            logger.error("copy_rates failed: %s", mt5.last_error())
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    # ── macro filter ─────────────────────────────────────────────────────────
    def _macro_filter(self, direction: str) -> bool:
        if not self.macro:
            return True

        usd_score = self.macro.get_usd_score()

        if direction == "sell" and usd_score <= 0:
            logger.info("Macro ALLOWS sell (USD_Score=%+d)", usd_score)
            return True
        elif direction == "buy" and usd_score >= 0:
            logger.info("Macro ALLOWS buy  (USD_Score=%+d)", usd_score)
            return True

        logger.info("Macro BLOCKS %s (USD_Score=%+d)", direction, usd_score)
        return False

    # ── position sizing ───────────────────────────────────────────────────────
    def _calc_volume(self, entry_price: float, stop_loss: float) -> float:
        """
        Расчёт объёма лота через tick_value с учётом доступной маржи.
        """
        acc = mt5.account_info()
        balance = acc.balance
        margin_free = acc.margin_free
        risk_money = balance * (self.config.RISK_PERCENT / 100.0)
        risk_dist = abs(entry_price - stop_loss)

        if risk_dist < 0.01:
            logger.warning("risk_dist=%.5f < 0.01 — слишком маленький, пропускаем", risk_dist)
            return 0.0

        si = mt5.symbol_info(self.config.SYMBOL)
        tick_size = si.trade_tick_size
        tick_value = si.trade_tick_value
        value_per_unit = tick_value / tick_size

        # Объём по риску
        volume_by_risk = risk_money / (risk_dist * value_per_unit)

        # Объём по марже (с запасом 20%)
        margin_per_lot = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY,
            self.config.SYMBOL,
            1.0,
            entry_price
        )

        if margin_per_lot and margin_per_lot > 0:
            volume_by_margin = (margin_free * 0.8) / margin_per_lot
            logger.info(
                "Margin check: free=%.2f | per_lot=%.2f | max_vol_by_margin=%.2f",
                margin_free, margin_per_lot, volume_by_margin
            )
        else:
            logger.warning("order_calc_margin failed, используем только расчёт по риску")
            volume_by_margin = volume_by_risk

        # Берём минимальный из двух
        volume_raw = min(volume_by_risk, volume_by_margin)

        # Округление
        step = si.volume_step
        volume = round(volume_raw / step) * step
        volume = max(si.volume_min, min(volume, si.volume_max))

        logger.info(
            "Vol sizing: balance=%.2f | risk_money=%.2f | dist=%.5f | "
            "vol_by_risk=%.2f | vol_by_margin=%.2f | final=%.2f",
            balance, risk_money, risk_dist,
            volume_by_risk, volume_by_margin, volume
        )
        return volume

    # ── open order ────────────────────────────────────────────────────────────
    def _open(self, signal: Dict) -> bool:
        # 1. Получаем свежие бары
        df_check = self._get_bars(100)
        if df_check is None:
            logger.warning("[GUARD] не удалось получить бары — отмена")
            return False

        # 2. Проверяем что IB сигнал всё ещё существует
        found, confirmed_signal = TechnicalIndicators.identify_inside_bar(df_check)
        if not found or not confirmed_signal:
            logger.warning("[GUARD] IB сигнал исчез — отмена")
            return False

        # 3. Проверяем направление не изменилось
        if confirmed_signal["direction"] != signal["direction"]:
            logger.warning("[GUARD] направление изменилось — отмена")
            return False

        # 4. НОВОЕ: проверяем что это тот же самый IB по времени свечи
        if confirmed_signal["inside_bar_time"] != signal["inside_bar_time"]:
            logger.warning("[GUARD] другой IB сигнал — отмена")
            return False

        pos = mt5.positions_get(symbol=self.config.SYMBOL)
        if pos and len(pos) >= self.config.MAX_POSITIONS:
            logger.info("Max positions (%d) reached", self.config.MAX_POSITIONS)
            return False

        # Макро-фильтр
        if not self._macro_filter(signal["direction"]):
            return False

        # Объём
        volume = self._calc_volume(signal["entry_price"], signal["stop_loss"])
        if volume <= 0:
            logger.error("Calculated volume <= 0, aborting")
            return False

        risk_dist = abs(signal["entry_price"] - signal["stop_loss"])
        price_info = mt5.symbol_info_tick(self.config.SYMBOL)

        if signal["direction"] == "buy":
            entry_price = price_info.ask
            sl = signal["stop_loss"]
            tp = signal["entry_price"] + risk_dist * self.config.RISK_REWARD
            otype = mt5.ORDER_TYPE_BUY
        else:
            entry_price = price_info.bid
            sl = signal["stop_loss"]
            tp = signal["entry_price"] - risk_dist * self.config.RISK_REWARD
            otype = mt5.ORDER_TYPE_SELL

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.SYMBOL,
            "volume": volume,
            "type": otype,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.config.MAGIC_NUMBER,
            "comment": f"IB_{signal['direction']}",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        logger.info(
            "ORDER REQUEST: %s %s | vol=%.2f | entry=%.5f | sl=%.5f | tp=%.5f",
            self.config.SYMBOL,
            "BUY" if otype == mt5.ORDER_TYPE_BUY else "SELL",
            volume, entry_price, sl, tp,
        )

        result = mt5.order_send(req)

        # ── обработка ошибок ──
        if result.retcode == 10027:
            logger.error("=== ERROR 10027: AutoTrading DISABLED ===")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "order_send FAILED | retcode=%d | comment=%s",
                result.retcode, result.comment,
            )
            return False

        # ── ВЕРИФИКАЦИЯ ФАКТИЧЕСКОГО ОБЪЁМА ──
        # Сразу после открытия проверяем, что реально открылось
        time.sleep(0.5)  # небольшая задержка для синхронизации
        positions = mt5.positions_get(ticket=result.order)

        if positions and len(positions) > 0:
            actual_volume = positions[0].volume
            if abs(actual_volume - volume) > 0.01:
                logger.warning(
                    "!!! ОБЪЁМ НЕ СОВПАДАЕТ !!! Запросили %.2f, открылось %.2f",
                    volume, actual_volume
                )
        else:
            logger.warning("Не удалось найти открытую позицию по ticket=%d", result.order)

        # ── успешно открыто ──
        usd_score = self.macro.get_usd_score() if self.macro else 0
        acc = mt5.account_info()
        risk_money = acc.balance * (self.config.RISK_PERCENT / 100.0)
        cur = TradingConfig.ACCOUNT_CURRENCY or "???"
        magic_number = self.config.MAGIC_NUMBER

        logger.info(
            "[TRADE OPENED] %s | ticket=%d | vol=%.2f | entry=%.5f | SL=%.5f | TP=%.5f | magic=%d | usd_score=%+d",
            signal["direction"].upper(), result.order, volume, entry_price, sl, tp, magic_number, usd_score,
        )

        trade_logger.info(
            "symbol=%s | dir=%s | lots=%.2f | risk=%.2f %s | usd_score=%+d | magic=%d | "
            "entry=%.5f | sl=%.5f | tp=%.5f",
            self.config.SYMBOL,
            signal["direction"].upper(),
            volume, risk_money, cur, usd_score, magic_number,
            entry_price, sl, tp,
        )

        self._notify(
            f"[TRADE]\n{self.config.SYMBOL} {signal['direction'].upper()}\n"
            f"Entry {entry_price:.5f} | SL {sl:.5f} | TP {tp:.5f}\n"
            f"Volume {volume} lots | {cur}"
        )
        return True

    # ── main check ────────────────────────────────────────────────────────────
    def check_and_trade(self):
        df = self._get_bars(100)
        if df is None:
            return

        found, signal = TechnicalIndicators.identify_inside_bar(df)

        if not found or not signal:
            return

        current_bar_time = df.index[-1]

        # Дедупликация
        if self._last_signal_bar_time == current_bar_time:
            logger.debug(
                "[SIGNAL ACTIVE] %s breakout bar [%s] ещё открыт",
                signal["direction"].upper(), current_bar_time,
            )
            return

        # Новый сигнал
        mother_bar = df.iloc[-3]
        inside_bar = df.iloc[-2]
        current_bar = df.iloc[-1]

        logger.info("=" * 60)
        logger.info("[SIGNAL DETECTED] Inside Bar Breakout — %s", signal["direction"].upper())
        logger.info(
            "  Mother  [%s]  H=%.5f  L=%.5f",
            mother_bar.name, mother_bar["high"], mother_bar["low"],
        )
        logger.info(
            "  Inside  [%s]  H=%.5f  L=%.5f",
            inside_bar.name, inside_bar["high"], inside_bar["low"],
        )
        logger.info(
            "  Break   [%s]  H=%.5f  L=%.5f",
            current_bar.name, current_bar["high"], current_bar["low"],
        )
        logger.info("=" * 60)

        self._last_signal_bar_time = current_bar_time
        self._open(signal)

# =============================================================================
# MONITOR
# =============================================================================

class PerformanceMonitor:
    @staticmethod
    def print_status(df: Optional[pd.DataFrame]):
        acc = mt5.account_info()
        pos = mt5.positions_get() or []
        cur = acc.currency

        logger.info("------------------------------------------------------------")
        logger.info("  Balance %.2f %s | Equity %.2f %s | P/L %.2f %s | Margin Free %.2f %s",
                    acc.balance, cur, acc.equity, cur, acc.profit, cur, acc.margin_free, cur)
        logger.info("  Margin Level %.2f%% | Open Positions %d",
                    acc.margin_level if acc.margin > 0 else 0.0, len(pos))

        for p in pos:
            logger.info("    #%d %s %.2f lots %s @ %.5f  P/L %.2f %s",
                        p.ticket,
                        "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                        p.volume, p.symbol, p.price_open, p.profit, cur)

        if df is not None and len(df) >= 3:
            mother_bar = df.iloc[-3]
            inside_bar = df.iloc[-2]
            current_bar = df.iloc[-1]
            logger.info(
                "IB CHECK:\n"
                "  Mother  [%s] H=%.5f L=%.5f\n"
                "  Inside  [%s] H=%.5f L=%.5f\n"
                "  Break   [%s] H=%.5f L=%.5f",
                mother_bar.name, mother_bar.high, mother_bar.low,
                inside_bar.name, inside_bar.high, inside_bar.low,
                current_bar.name, current_bar.high, current_bar.low,
            )

        logger.info("------------------------------------------------------------")


# =============================================================================
# BOT
# =============================================================================

class MT5TradingBot:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.trader = InsideBarTrader(config)
        self.running = False

    def start(self):
        logger.info("============================================================")
        logger.info("  MT5 TRADING BOT  —  starting")
        logger.info("============================================================")

        if not MT5Connection.initialize():
            return
        if not MT5Connection.login():
            MT5Connection.shutdown()
            return

        si = mt5.symbol_info(self.config.SYMBOL)
        if si is None:
            logger.error("Symbol '%s' not found", self.config.SYMBOL)
            MT5Connection.shutdown()
            return
        if not si.visible:
            mt5.symbol_select(self.config.SYMBOL, True)

        logger.info("  Symbol        : %s", self.config.SYMBOL)
        logger.info("  Timeframe     : H4")
        logger.info("  Risk          : %.2f%%", self.config.RISK_PERCENT)
        logger.info("  R:R           : 1:%.2f", self.config.RISK_REWARD)
        logger.info("  Macro filter  : %s", "ON" if self.config.USE_MACRO_FILTER else "OFF")
        logger.info("  Vol min/max   : %.2f / %.2f  step=%.2f",
                    si.volume_min, si.volume_max, si.volume_step)
        logger.info("")
        logger.info("  >>> Make sure AutoTrading is ENABLED in MT5 toolbar <<<")
        logger.info("")

        self.running = True
        last_status_log = time.time()

        try:
            while self.running:
                try:
                    h = datetime.now().hour
                    if not (self.config.TRADING_START_HOUR <= h <= self.config.TRADING_END_HOUR):
                        logger.info("Outside trading hours, sleeping 60s ...")
                        time.sleep(60)
                        continue

                    df = self.trader._get_bars(100)

                    if time.time() - last_status_log >= self.config.STATUS_LOG_INTERVAL:
                        PerformanceMonitor.print_status(df)
                        last_status_log = time.time()

                    self.trader.check_and_trade()
                    time.sleep(self.config.CHECK_INTERVAL)

                except Exception as exc:
                    logger.error("Main loop error: %s", exc, exc_info=True)
                    time.sleep(60)

        except KeyboardInterrupt:
            logger.info("[STOP] Ctrl+C received")
        finally:
            self.running = False
            MT5Connection.shutdown()
            logger.info("Bot stopped.")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    config = TradingConfig()
    bot = MT5TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()
