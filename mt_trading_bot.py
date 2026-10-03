"""
MetaTrader 5 Live Trading Bot
Автоматическая торговля с Inside Bar стратегией + макро-фильтром
Готово для развертывания на AWS
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Tuple, List
import json
import os
from fredapi import Fred


# =============================================================================
# CONFIGURATION
# =============================================================================

class TradingConfig:
    """Конфигурация бота"""

    # MT5 Connection
    MT5_LOGIN = int(os.getenv('MT5_LOGIN', '5045893454'))
    MT5_PASSWORD = os.getenv('MT5_PASSWORD', '1_IhKrTj')
    MT5_SERVER = os.getenv('MT5_SERVER', 'OANDATMS-MT5')

    # Trading Parameters
    SYMBOL = os.getenv('SYMBOL', 'XAUUSD')  # Brent Oil
    TIMEFRAME = mt5.TIMEFRAME_H4  # 4-hour
    MAGIC_NUMBER = 123456  # Уникальный идентификатор бота

    # Risk Management
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '10000'))
    RISK_PERCENT = float(os.getenv('RISK_PERCENT', '1.0'))
    RISK_REWARD = float(os.getenv('RISK_REWARD', '2.75'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '1'))

    # Strategy Parameters
    ATR_PERIOD = 14
    MIN_ATR = 0.5  # Минимальный ATR для фильтрации низкой волатильности

    # Macro Filter
    FRED_API_KEY = os.getenv('FRED_API_KEY', 'bb21719b26aff61c740702ea701a07ed')
    USE_MACRO_FILTER = os.getenv('USE_MACRO_FILTER', 'true').lower() == 'true'

    # Trading Hours (UTC)
    TRADING_START_HOUR = 0
    TRADING_END_HOUR = 23

    # Bot Settings
    CHECK_INTERVAL = 300  # Проверка каждые 5 минут
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')

    # AWS/Cloud
    ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8369050793:AAF5KnCZwvR4TjjvfAx4GcKyxBzorM_FojQ')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'mt5oandatestbot')


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=getattr(logging, TradingConfig.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(TradingConfig.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('MT5_TradingBot')


logger = setup_logging()


# =============================================================================
# MT5 CONNECTION MANAGER
# =============================================================================

class MT5Connection:
    """Управление подключением к MT5"""

    @staticmethod
    def initialize() -> bool:
        """Инициализация MT5"""
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        logger.info(f"MT5 version: {mt5.version()}")
        return True

    @staticmethod
    def login() -> bool:
        """Авторизация в MT5"""
        authorized = mt5.login(
            login=TradingConfig.MT5_LOGIN,
            password=TradingConfig.MT5_PASSWORD,
            server=TradingConfig.MT5_SERVER
        )

        if not authorized:
            logger.error(f"Login failed: {mt5.last_error()}")
            return False

        account_info = mt5.account_info()
        logger.info(f"Connected to account #{account_info.login}")
        logger.info(f"Balance: ${account_info.balance:.2f}")
        logger.info(f"Equity: ${account_info.equity:.2f}")

        return True

    @staticmethod
    def shutdown():
        """Закрытие соединения"""
        mt5.shutdown()
        logger.info("MT5 connection closed")


# =============================================================================
# MACRO DATA PROCESSOR
# =============================================================================

class MacroDataProcessor:
    """Обработка макроэкономических данных"""

    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key) if api_key else None
        self.cache = {}
        self.last_update = None

    def get_usd_score(self) -> float:
        """
        Получить текущий USD Score
        Кэшируется на 1 день (макроданные обновляются редко)
        """
        if not self.fred:
            logger.warning("FRED API key not set, macro filter disabled")
            return 0

        # Проверка кэша
        if self.last_update and (datetime.now() - self.last_update).days < 1:
            return self.cache.get('usd_score', 0)

        try:
            logger.info("Fetching macro data from FRED...")

            # Получаем последние данные
            indicators = {
                'PAYEMS': 'NFP_Total_Jobs',
                'DFII10': 'TIPS_10Y_Real_Yield',
                'BOPGSTB': 'Trade_Balance',
            }

            usd_score = 0

            for ticker, name in indicators.items():
                try:
                    series = self.fred.get_series(ticker, observation_start=datetime.now() - timedelta(days=90))

                    if len(series) >= 2:
                        # Рассчитываем изменение
                        change = series.iloc[-1] - series.iloc[-2]

                        # NFP и Trade Balance: рост = сильный USD
                        if ticker in ['PAYEMS', 'BOPGSTB']:
                            usd_score += 1 if change > 0 else -1
                        # TIPS: рост = сильный USD
                        elif ticker == 'DFII10':
                            usd_score += 1 if change > 0 else -1

                except Exception as e:
                    logger.warning(f"Failed to fetch {ticker}: {e}")

            # Кэшируем
            self.cache['usd_score'] = usd_score
            self.last_update = datetime.now()

            logger.info(f"USD Score updated: {usd_score}")
            return usd_score

        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            return 0


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicators:
    """Technical Indicators"""

    @staticmethod
    def identify_inside_bar(df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """
        Определение Inside Bar и направления пробоя

        Returns:
            (is_setup, signal_dict)
        """
        if len(df) < 3:
            return False, None

        i = len(df) - 1

        # Предыдущий бар (потенциальный Inside Bar)
        prev_bar = df.iloc[-2]
        # Бар до него (Mother Bar)
        mother_bar = df.iloc[-3]
        # Текущий бар
        current_bar = df.iloc[i]

        # Проверка Inside Bar
        is_inside = (prev_bar['high'] < mother_bar['high']) and \
                    (prev_bar['low'] > mother_bar['low'])

        if not is_inside:
            return False, None

        # Проверка пробоя на текущем баре
        ib_high = prev_bar['high']
        ib_low = prev_bar['low']

        signal = None

        if current_bar['high'] > ib_high:
            # Пробой вверх
            signal = {
                'direction': 'buy',
                'entry_price': ib_high,
                'stop_loss': ib_low,
                'inside_bar_time': prev_bar.name
            }
        elif current_bar['low'] < ib_low:
            # Пробой вниз
            signal = {
                'direction': 'sell',
                'entry_price': ib_low,
                'stop_loss': ib_high,
                'inside_bar_time': prev_bar.name
            }

        return True, signal


# =============================================================================
# TRADING LOGIC
# =============================================================================

class InsideBarTrader:
    """Основная торговая логика"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.macro_processor = MacroDataProcessor(config.FRED_API_KEY) if config.USE_MACRO_FILTER else None

    def get_market_data(self, bars: int = 100) -> Optional[pd.DataFrame]:
        """Получить рыночные данные"""
        rates = mt5.copy_rates_from_pos(
            self.config.SYMBOL,
            self.config.TIMEFRAME,
            0,
            bars
        )

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get market data: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def check_macro_filter(self, direction: str) -> bool:
        """Проверка макро-фильтра"""
        if not self.macro_processor:
            return True

        usd_score = self.macro_processor.get_usd_score()

        # Для BRENT (нефть):
        # Сильный USD (positive score) -> предпочитаем Sell
        # Слабый USD (negative score) -> предпочитаем Buy

        if direction == 'sell' and usd_score <= 0:
            return True
        elif direction == 'buy' and usd_score >= 0:
            return True

        logger.info(f"Trade filtered by macro: {direction} with USD_Score={usd_score}")
        return False

    def calculate_position_size(self, entry: float, stop_loss: float) -> float:
        """
        Расчет размера позиции на основе риска

        Returns:
            Лоты
        """
        account_info = mt5.account_info()
        balance = account_info.balance

        risk_money = balance * (self.config.RISK_PERCENT / 100)
        risk_points = abs(entry - stop_loss)

        # Получаем информацию о символе
        symbol_info = mt5.symbol_info(self.config.SYMBOL)
        if not symbol_info:
            logger.error(f"Symbol {self.config.SYMBOL} not found")
            return 0

        point_value = symbol_info.trade_contract_size * symbol_info.point

        # Расчет объема
        volume = risk_money / (risk_points * point_value / symbol_info.point)

        # Округление до допустимого шага
        volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step

        # Проверка лимитов
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))

        logger.info(f"Calculated position size: {volume} lots (Risk: ${risk_money:.2f})")
        return volume

    def open_position(self, signal: Dict, atr: float) -> bool:
        """Открытие позиции"""

        # Проверка количества открытых позиций
        positions = mt5.positions_get(symbol=self.config.SYMBOL)
        if positions and len(positions) >= self.config.MAX_POSITIONS:
            logger.info("Max positions reached")
            return False

        # Проверка макро-фильтра
        if not self.check_macro_filter(signal['direction']):
            return False

        # Расчет размера позиции
        volume = self.calculate_position_size(signal['entry_price'], signal['stop_loss'])

        if volume <= 0:
            logger.error("Invalid position size")
            return False

        # Расчет TP
        risk_dist = abs(signal['entry_price'] - signal['stop_loss'])

        if signal['direction'] == 'buy':
            tp = signal['entry_price'] + (risk_dist * self.config.RISK_REWARD)
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.config.SYMBOL).ask
        else:
            tp = signal['entry_price'] - (risk_dist * self.config.RISK_REWARD)
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.config.SYMBOL).bid

        # Подготовка запроса
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.SYMBOL,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": signal['stop_loss'],
            "tp": tp,
            "deviation": 20,
            "magic": self.config.MAGIC_NUMBER,
            "comment": f"InsideBar_{signal['direction']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        # Отправка ордера
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return False

        logger.info(f"  Position opened: {signal['direction'].upper()}")
        logger.info(f"   Volume: {volume} lots")
        logger.info(f"   Entry: {price:.5f}")
        logger.info(f"   SL: {signal['stop_loss']:.5f}")
        logger.info(f"   TP: {tp:.5f}")
        logger.info(f"   Ticket: {result.order}")

        # Отправка уведомления
        self.send_notification(
            f"🔔 NEW TRADE\n"
            f"Symbol: {self.config.SYMBOL}\n"
            f"Direction: {signal['direction'].upper()}\n"
            f"Entry: {price:.5f}\n"
            f"SL: {signal['stop_loss']:.5f}\n"
            f"TP: {tp:.5f}\n"
            f"Volume: {volume} lots"
        )

        return True

    def check_and_trade(self):
        """Основной цикл проверки и торговли"""
        logger.info("Checking for trading opportunities...")

        # Получить данные
        df = self.get_market_data(100)
        if df is None:
            return

        # Рассчитать ATR
        df['atr'] = TechnicalIndicators.calculate_atr(df, self.config.ATR_PERIOD)
        current_atr = df['atr'].iloc[-1]

        # Фильтр волатильности
        if current_atr < self.config.MIN_ATR:
            logger.info(f"Low volatility (ATR={current_atr:.5f}), skipping")
            return

        # Проверка Inside Bar
        is_setup, signal = TechnicalIndicators.identify_inside_bar(df)

        if is_setup and signal:
            logger.info(f"📊 Inside Bar setup detected: {signal['direction'].upper()}")
            self.open_position(signal, current_atr)
        else:
            logger.debug("No trading setup")

    def send_notification(self, message: str):
        """Отправка уведомлений (Telegram)"""
        if not self.config.ENABLE_NOTIFICATIONS:
            return

        try:
            import requests
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": self.config.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")


# =============================================================================
# MONITORING & REPORTING
# =============================================================================

class PerformanceMonitor:
    """Мониторинг производительности"""

    @staticmethod
    def get_account_stats() -> Dict:
        """Статистика счета"""
        account_info = mt5.account_info()

        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'margin_free': account_info.margin_free,
            'margin_level': account_info.margin_level
        }

    @staticmethod
    def get_open_positions() -> List[Dict]:
        """Открытые позиции"""
        positions = mt5.positions_get()

        if not positions:
            return []

        return [{
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': pos.volume,
            'price_open': pos.price_open,
            'price_current': pos.price_current,
            'sl': pos.sl,
            'tp': pos.tp,
            'profit': pos.profit,
            'comment': pos.comment
        } for pos in positions]

    @staticmethod
    def get_trade_history(days: int = 7) -> pd.DataFrame:
        """История сделок"""
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()

        deals = mt5.history_deals_get(from_date, to_date)

        if not deals:
            return pd.DataFrame()

        df = pd.DataFrame([{
            'ticket': deal.ticket,
            'time': datetime.fromtimestamp(deal.time),
            'symbol': deal.symbol,
            'type': deal.type,
            'volume': deal.volume,
            'price': deal.price,
            'profit': deal.profit,
            'commission': deal.commission,
            'comment': deal.comment
        } for deal in deals])

        return df

    @staticmethod
    def print_status():
        """Вывод текущего статуса"""
        stats = PerformanceMonitor.get_account_stats()
        positions = PerformanceMonitor.get_open_positions()

        logger.info("=" * 80)
        logger.info("ACCOUNT STATUS")
        logger.info("=" * 80)
        logger.info(f"Balance:      ${stats['balance']:,.2f}")
        logger.info(f"Equity:       ${stats['equity']:,.2f}")
        logger.info(f"Profit:       ${stats['profit']:,.2f}")
        logger.info(f"Margin Level: {stats['margin_level']:.2f}%")
        logger.info(f"Margin Free:  ${stats['margin_free']:,.2f}")
        logger.info(f"Open Positions: {len(positions)}")

        if positions:
            logger.info("\nOPEN POSITIONS:")
            for pos in positions:
                logger.info(
                    f"  #{pos['ticket']} {pos['type']} {pos['volume']} {pos['symbol']} @ {pos['price_open']} | P/L: ${pos['profit']:.2f}")

        logger.info("=" * 80)


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class MT5TradingBot:
    """Главный класс бота"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.trader = InsideBarTrader(config)
        self.running = False

    def start(self):
        """Запуск бота"""
        logger.info("=" * 80)
        logger.info("MT5 TRADING BOT STARTING")
        logger.info("=" * 80)

        # Инициализация MT5
        if not MT5Connection.initialize():
            return

        # Авторизация
        if not MT5Connection.login():
            MT5Connection.shutdown()
            return

        # Проверка символа
        symbol_info = mt5.symbol_info(self.config.SYMBOL)
        if symbol_info is None:
            logger.error(f"Symbol {self.config.SYMBOL} not found")
            MT5Connection.shutdown()
            return

        if not symbol_info.visible:
            if not mt5.symbol_select(self.config.SYMBOL, True):
                logger.error(f"Failed to select {self.config.SYMBOL}")
                MT5Connection.shutdown()
                return

        logger.info(f"Trading symbol: {self.config.SYMBOL}")
        logger.info(f"Timeframe: {self._timeframe_to_str(self.config.TIMEFRAME)}")
        logger.info(f"Risk per trade: {self.config.RISK_PERCENT}%")
        logger.info(f"Risk:Reward: 1:{self.config.RISK_REWARD}")
        logger.info(f"Macro filter: {'Enabled' if self.config.USE_MACRO_FILTER else 'Disabled'}")

        # Главный цикл
        self.running = True
        logger.info("\n🚀 Bot is now running...\n")

        try:
            while self.running:
                try:
                    # Проверка торговых часов
                    current_hour = datetime.now().hour
                    if not (self.config.TRADING_START_HOUR <= current_hour <= self.config.TRADING_END_HOUR):
                        logger.debug("Outside trading hours")
                        time.sleep(60)
                        continue

                    # Проверка статуса
                    PerformanceMonitor.print_status()

                    # Проверка сигналов
                    self.trader.check_and_trade()

                    # Ожидание
                    logger.info(f"Next check in {self.config.CHECK_INTERVAL} seconds...")
                    time.sleep(self.config.CHECK_INTERVAL)

                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    time.sleep(60)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Bot stopped by user")

        finally:
            self.stop()

    def stop(self):
        """Остановка бота"""
        logger.info("Stopping bot...")
        self.running = False
        MT5Connection.shutdown()
        logger.info("Bot stopped")

    @staticmethod
    def _timeframe_to_str(timeframe) -> str:
        """Конвертация timeframe в строку"""
        tf_map = {
            mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15", mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1"
        }
        return tf_map.get(timeframe, "Unknown")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Точка входа"""
    config = TradingConfig()
    bot = MT5TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()