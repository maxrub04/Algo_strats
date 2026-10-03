"""
MetaTrader 5 Live Trading Bot
Inside Bar стратегия + макро-фильтр (FRED)
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
    MT5_LOGIN    = int(os.getenv("MT5_LOGIN",    "62638199"))
    MT5_PASSWORD =     os.getenv("MT5_PASSWORD", "nEdgos-refkeg1")
    MT5_SERVER   =     os.getenv("MT5_SERVER",   "OANDATMS-MT5")

    SYMBOL    = os.getenv("SYMBOL",    "OILBRNT.pro")
    TIMEFRAME = mt5.TIMEFRAME_M1
    MAGIC_NUMBER = 1234

    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "50000"))   # PLN
    RISK_PERCENT    = float(os.getenv("RISK_PERCENT",    "1.0"))
    RISK_REWARD     = float(os.getenv("RISK_REWARD",     "2.75"))
    MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS",     "3"))

    ACCOUNT_CURRENCY = ""

    ATR_PERIOD = 14
    MIN_ATR    = 0.00001

    FRED_API_KEY     = os.getenv("FRED_API_KEY",     "bb21719b26aff61c740702ea701a07ed")
    USE_MACRO_FILTER = os.getenv("USE_MACRO_FILTER", "true").lower() == "true"

    TRADING_START_HOUR = 2
    TRADING_END_HOUR   = 23

    CHECK_INTERVAL      = 1    # проверка каждую секунду
    STATUS_LOG_INTERVAL = 600  # секунды
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE  = os.getenv("LOG_FILE",  "trading_bot.log")

    ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
    TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "8369050793:AAF5KnCZwvR4TjjvfAx4GcKyxBzorM_FojQ")
    TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID",   "mt5oandatestbot")


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    fmt   = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    _CACHE_TTL_SEC = 86400   # 24 часа

    def __init__(self, api_key: str):
        self.fred: Optional[Fred] = None
        self.cache: Dict = {}
        self.last_update: Optional[datetime] = None

        if api_key:
            try:
                self.fred = Fred(api_key=api_key)
                logger.info("FRED client ready")
            except Exception as exc:
                logger.error("FRED init error: %s", exc)

    def get_usd_score(self) -> int:
        if not self.fred:
            return 0

        if (self.last_update and
                (datetime.now() - self.last_update).total_seconds() < self._CACHE_TTL_SEC):
            return self.cache.get("usd_score", 0)

        logger.info("Fetching macro data from FRED ...")
        score   = 0
        fetched = 0

        tickers = {
            "PAYEMS":  "NFP",
            "DFII10":  "TIPS_10Y",
            "BOPGSTB": "Trade_Balance",
            "EXPGS":   "Exports",
            "IMPGS":   "Imports",
        }

        for ticker, label in tickers.items():
            try:
                series = self.fred.get_series(
                    ticker,
                    observation_start=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                )
                if series is not None and len(series) >= 2:
                    change  = series.iloc[-1] - series.iloc[-2]
                    contrib = 1 if change > 0 else -1
                    score  += contrib
                    fetched += 1
                    logger.info("  %s: change=%.4f  contrib=%+d", label, change, contrib)
                else:
                    logger.warning("  %s: not enough data", label)
            except Exception as exc:
                logger.warning("  %s fetch error: %s", label, exc)

        if fetched == 0:
            logger.warning("No FRED indicators fetched — using neutral 0")
            score = 0

        self.cache["usd_score"] = score
        self.last_update = datetime.now()
        logger.info("USD Score = %d  (%d/%d indicators fetched)", score, fetched, len(tickers))
        return score


# =============================================================================
# TECHNICALS
# =============================================================================

class TechnicalIndicators:

    @staticmethod
    def identify_inside_bar(df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """
        Паттерн Inside Bar:
          df.iloc[-3] = Mother Bar
          df.iloc[-2] = Inside Bar  (high < mother.high AND low > mother.low)
          df.iloc[-1] = Current Bar (breakout candle)
        """
        if len(df) < 3:
            return False, None

        mother_bar  = df.iloc[-3]
        inside_bar  = df.iloc[-2]
        current_bar = df.iloc[-1]

        # Проверяем, что -2 это реально Inside Bar
        is_inside = (
            inside_bar["high"] <= mother_bar["high"] and
            inside_bar["low"]  >= mother_bar["low"]
        )
        if not is_inside:
            return False, None

        ib_high = inside_bar["high"]
        ib_low  = inside_bar["low"]
        signal  = {}

        if current_bar["high"] > ib_high:
            signal = {
                "direction":      "buy",
                "entry_price":    ib_high,
                "stop_loss":      ib_low,
                "inside_bar_time": inside_bar.name,
            }
        elif current_bar["low"] < ib_low:
            signal = {
                "direction":      "sell",
                "entry_price":    ib_low,
                "stop_loss":      ib_high,
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
        self.macro  = MacroDataProcessor(config.FRED_API_KEY) if config.USE_MACRO_FILTER else None
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
        Расчёт объёма лота по формуле:
            volume = risk_money / (risk_dist * value_per_unit)
        где value_per_unit = tick_value / tick_size

        OILBRNT.pro:
            tick_size  = 0.01
            tick_value = 35.347  (PLN за 1 тик на 1 лот)
            → value_per_unit = 3534.7

        Пример: balance=50000, risk=1%, dist=0.50
            volume = 500 / (0.50 * 3534.7) = 0.28 лота ✅
        """
        acc        = mt5.account_info()
        balance    = acc.balance
        margin_free  = acc.margin_free
        risk_money = balance * (self.config.RISK_PERCENT / 100.0)
        risk_dist  = abs(entry_price - stop_loss)

        if risk_dist < 0.01:
            logger.warning("risk_dist=%.5f < 0.01 (1 тик) — слишком маленький IB, пропускаем", risk_dist)
            return 0.0

        si         = mt5.symbol_info(self.config.SYMBOL)
        tick_size  = si.trade_tick_size    # 0.01
        tick_value = si.trade_tick_value   # 35.347

        value_per_unit = tick_value / tick_size    # 3534.7

        volume_raw = risk_money / (risk_dist * value_per_unit)

        margin_per_lot = si.margin_initial or mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY,  # тип значения не важен для расчета маржи
            self.config.SYMBOL,
            1.0,  # объем = 1 лот
            entry_price
        )

        if margin_per_lot:
            # Максимальный объем по марже с запасом 5%
            volume_by_margin = (margin_free * 0.95) / margin_per_lot
            # Берем минимальный из двух объемов
            volume_raw = min(volume_raw, volume_by_margin)
        else:
            logger.warning("Не удалось получить маржинальные требования, используем только расчет по риску")
            volume_raw = volume_raw

        step   = si.volume_step    # 0.01
        volume = round(round(volume_raw / step) * step, 2)


        if volume < si.volume_min:
            logger.warning(
                "Calculated vol=%.4f < volume_min=%.2f — using volume_min",
                volume, si.volume_min
            )
            volume = si.volume_min

        if volume > si.volume_max:
            logger.warning(
                "Calculated vol=%.4f > volume_max=%.2f — capping to volume_max",
                volume, si.volume_max
            )
            volume = si.volume_max

        logger.info(
            "Vol sizing: balance=%.2f | risk_money=%.2f | dist=%.5f | "
            "tick_size=%.4f | tick_val=%.4f | val_per_unit=%.2f | "
            "raw_vol=%.4f | final_vol=%.2f",
            balance, risk_money, risk_dist,
            tick_size, tick_value, value_per_unit,
            volume_raw, volume,
        )
        return volume

    # ── open order ────────────────────────────────────────────────────────────
    def _open(self, signal: Dict) -> bool:

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

        risk_dist  = abs(signal["entry_price"] - signal["stop_loss"])
        price_info = mt5.symbol_info_tick(self.config.SYMBOL)

        if signal["direction"] == "buy":
            entry_price = price_info.ask
            sl    = signal["stop_loss"]
            tp    = signal["entry_price"] + risk_dist * self.config.RISK_REWARD
            otype = mt5.ORDER_TYPE_BUY
        else:
            entry_price = price_info.bid
            sl    = signal["stop_loss"]
            tp    = signal["entry_price"] - risk_dist * self.config.RISK_REWARD
            otype = mt5.ORDER_TYPE_SELL

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       self.config.SYMBOL,
            "volume":       volume,
            "type":         otype,
            "price":        entry_price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,
            "magic":        self.config.MAGIC_NUMBER,
            "comment":      f"IB_{signal['direction']}",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        # ✅ Лог запроса перед отправкой — для дебага
        logger.info(
            "ORDER REQUEST: %s %s | vol=%.2f | entry=%.5f | sl=%.5f | tp=%.5f | dist=%.5f | RR=%.2f",
            self.config.SYMBOL,
            "BUY" if otype == mt5.ORDER_TYPE_BUY else "SELL",
            volume, entry_price, sl, tp, risk_dist, self.config.RISK_REWARD,
        )

        result = mt5.order_send(req)

        # ── обработка ошибок ──
        if result.retcode == 10027:
            logger.error("=== ERROR 10027: AutoTrading DISABLED ===")
            logger.error("  Включи AutoTrading в тулбаре MT5 (зелёная кнопка 'Auto')")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "order_send FAILED | retcode=%d | comment=%s | request=%s",
                result.retcode, result.comment, req,
            )
            return False

        # ── успешно открыто ──
        usd_score  = self.macro.get_usd_score() if self.macro else 0
        acc        = mt5.account_info()
        risk_money = acc.balance * (self.config.RISK_PERCENT / 100.0)
        cur        = TradingConfig.ACCOUNT_CURRENCY or "???"

        logger.info(
            "[TRADE OPENED] %s | ticket=%d | vol=%.2f | magic_number=% | entry=%.5f | SL=%.5f | TP=%.5f",
            signal["direction"].upper(), result.order, volume,self.config.MAGIC_NUMBER, entry_price, sl, tp,
        )

        trade_logger.info(
            "symbol=%s | dir=%s | lots=%.2f | risk=%.2f %s | usd_score=%+d | "
            "entry=%.5f | sl=%.5f | tp=%.5f",
            self.config.SYMBOL,
            signal["direction"].upper(),
            volume, risk_money, cur, usd_score,
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

        # Время текущей (breakout) свечи — последняя строка датафрейма
        current_bar_time = df.index[-1]

        # ── Дедупликация ──────────────────────────────────────────────────────
        # Каждую секунду бот видит ту же открытую свечу.
        # Если мы уже обработали этот бар — просто логируем "ещё открыт" и выходим.
        if self._last_signal_bar_time == current_bar_time:
            logger.debug(
                "[SIGNAL ACTIVE] %s breakout bar [%s] ещё открыт, ждём следующей свечи",
                signal["direction"].upper(), current_bar_time,
            )
            return

        # ── Новый сигнал — логируем полностью ────────────────────────────────
        mother_bar  = df.iloc[-3]
        inside_bar  = df.iloc[-2]
        current_bar = df.iloc[-1]

        logger.info("=" * 60)
        logger.info("[SIGNAL DETECTED] Inside Bar Breakout — %s", signal["direction"].upper())
        logger.info(
            "  Mother  [%s]  H=%.5f  L=%.5f",
            mother_bar.name, mother_bar["high"], mother_bar["low"],
        )
        logger.info(
            "  Inside  [%s]  H=%.5f  L=%.5f  (IB high=%.5f  low=%.5f)",
            inside_bar.name, inside_bar["high"], inside_bar["low"],
            signal["entry_price"], signal["stop_loss"],
        )
        logger.info(
            "  Break   [%s]  H=%.5f  L=%.5f  -> прорыв %s",
            current_bar.name, current_bar["high"], current_bar["low"],
            "ВВЕРХ" if signal["direction"] == "buy" else "ВНИЗ",
        )
        logger.info(
            "  Entry=%.5f  SL=%.5f  Risk dist=%.5f",
            signal["entry_price"], signal["stop_loss"],
            abs(signal["entry_price"] - signal["stop_loss"]),
        )
        logger.info("=" * 60)

        # Запоминаем этот бар — повторно не входим пока свеча не закроется
        self._last_signal_bar_time = current_bar_time

        # Пытаемся открыть ордер
        self._open(signal)

    # ── telegram ──────────────────────────────────────────────────────────────
    def _notify(self, text: str):
        if not self.config.ENABLE_NOTIFICATIONS:
            return
        try:
            import requests
            requests.post(
                f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage",
                data={"chat_id": self.config.TELEGRAM_CHAT_ID, "text": text},
                timeout=10,
            )
        except Exception as exc:
            logger.warning("Telegram failed: %s", exc)


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
        logger.info("  Balance %.2f %s | Equity %.2f %s | P/L %.2f %s",
                    acc.balance, cur, acc.equity, cur, acc.profit, cur)
        logger.info("  Margin Level %.2f%% | Open Positions %d",
                    acc.margin_level if acc.margin > 0 else 0.0, len(pos))

        for p in pos:
            logger.info("    #%d %s %.2f %s @ %.5f  P/L %.2f %s",
                        p.ticket,
                        "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                        p.volume, p.symbol, p.price_open, p.profit, cur)

        if df is not None and len(df) >= 3:
            mother_bar  = df.iloc[-3]
            inside_bar  = df.iloc[-2]
            current_bar = df.iloc[-1]
            logger.info(
                "IB CHECK:\n"
                "  Mother  [%s] O=%.5f H=%.5f L=%.5f C=%.5f\n"
                "  Inside  [%s] O=%.5f H=%.5f L=%.5f C=%.5f\n"
                "  Break   [%s] O=%.5f H=%.5f L=%.5f C=%.5f",
                mother_bar.name,  mother_bar.open,  mother_bar.high,  mother_bar.low,  mother_bar.close,
                inside_bar.name,  inside_bar.open,  inside_bar.high,  inside_bar.low,  inside_bar.close,
                current_bar.name, current_bar.open, current_bar.high, current_bar.low, current_bar.close,
            )

        logger.info("------------------------------------------------------------")


# =============================================================================
# BOT
# =============================================================================

class MT5TradingBot:
    def __init__(self, config: TradingConfig):
        self.config  = config
        self.trader  = InsideBarTrader(config)
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
        logger.info("  Timeframe     : M1 (inside bar on last 3 bars)")
        logger.info("  Risk          : %.1f%%", self.config.RISK_PERCENT)
        logger.info("  R:R           : 1:%.2f", self.config.RISK_REWARD)
        logger.info("  Macro filter  : %s", "ON" if self.config.USE_MACRO_FILTER else "OFF")
        logger.info("  Vol min/max   : %.2f / %.2f  step=%.2f",
                    si.volume_min, si.volume_max, si.volume_step)
        logger.info("  Tick sz/val   : %.4f / %.4f",
                    si.trade_tick_size, si.trade_tick_value)
        logger.info("")
        logger.info("  >>> Make sure AutoTrading is ENABLED in MT5 toolbar <<<")
        logger.info("")

        self.running     = True
        last_status_log  = time.time()

        try:
            while self.running:
                try:
                    h = datetime.now().hour
                    if not (self.config.TRADING_START_HOUR <= h <= self.config.TRADING_END_HOUR):
                        logger.info("Outside trading hours (%d-%d), sleeping 60s ...",
                                    self.config.TRADING_START_HOUR, self.config.TRADING_END_HOUR)
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
    bot    = MT5TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()
