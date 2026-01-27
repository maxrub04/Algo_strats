"""
IBKR Live Trading Integration for Inside Bar Strategy
Добавьте этот код в ваш существующий файл или используйте как отдельный модуль
"""

from ib_insync import *
import pandas as pd
import time
from datetime import datetime, timedelta
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ibkr_trading.log'),
        logging.StreamHandler()
    ]
)


class IBKRTradingBot:
    """
    Класс для автоматической торговли через IBKR
    """

    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Args:
            host: IP адрес (обычно localhost)
            port: 7497 для paper trading, 7496 для live
            client_id: уникальный ID клиента
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.active_orders = {}
        self.positions = {}

    def connect(self):
        """Подключение к TWS/Gateway"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logging.info(f"✅ Connected to IBKR on port {self.port}")

            # Подписка на события
            self.ib.orderStatusEvent += self.on_order_status
            self.ib.execDetailsEvent += self.on_execution

            return True
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            return False

    def disconnect(self):
        """Отключение"""
        self.ib.disconnect()
        logging.info("Disconnected from IBKR")

    def create_contract(self, symbol, sec_type='CMDTY', exchange='SMART', currency='USD'):
        """
        Создание контракта для торговли

        Args:
            symbol: Тикер (например, 'XAUUSD', 'BRN' для нефти)
            sec_type: Тип инструмента ('CMDTY', 'CASH', 'STK')
            exchange: Биржа
            currency: Валюта
        """
        if sec_type == 'CMDTY':
            # Для товаров используем Forex-контракт или Futures
            if symbol == 'BRENTCMDUSD':
                # Brent Crude Oil Futures
                contract = ContFuture('BZ', exchange='NYMEX', currency='USD')
            elif symbol == 'XAUUSD':
                # Gold Forex
                contract = Forex('XAUUSD')
            else:
                contract = Contract(
                    symbol=symbol,
                    secType=sec_type,
                    exchange=exchange,
                    currency=currency
                )
        elif sec_type == 'CASH':
            # Forex
            contract = Forex(symbol)
        elif sec_type == 'STK':
            # Акции
            contract = Stock(symbol, exchange, currency)
        else:
            contract = Contract(
                symbol=symbol,
                secType=sec_type,
                exchange=exchange,
                currency=currency
            )

        # Квалификация контракта
        self.ib.qualifyContracts(contract)
        logging.info(f"Contract created: {contract}")
        return contract

    def get_current_price(self, contract):
        """Получение текущей цены"""
        ticker = self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(2)  # Ждем данные

        if ticker.marketPrice():
            return ticker.marketPrice()
        elif ticker.last:
            return ticker.last
        elif ticker.close:
            return ticker.close
        else:
            logging.warning("No price data available")
            return None

    def get_historical_data(self, contract, duration='1 D', bar_size='4 hours'):
        """
        Получение исторических данных

        Args:
            duration: '1 D', '5 D', '1 W', '1 M' и т.д.
            bar_size: '1 min', '5 mins', '1 hour', '4 hours', '1 day'
        """
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=False,  # False для 24/7 рынков
            formatDate=1
        )

        if bars:
            df = util.df(bars)
            df.rename(columns={
                'date': 'DateTime',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Vol'
            }, inplace=True)
            return df
        else:
            logging.warning("No historical data received")
            return pd.DataFrame()

    def calculate_position_size(self, entry_price, stop_loss, risk_percent, account_balance):
        """
        Расчет размера позиции на основе риска
        """
        risk_amount = account_balance * (risk_percent / 100.0)
        risk_distance = abs(entry_price - stop_loss)

        if risk_distance == 0:
            return 0

        position_size = risk_amount / risk_distance
        return int(position_size)  # Округляем до целых лотов

    def place_bracket_order(self, contract, action, quantity, entry_price,
                            stop_loss, take_profit, order_type='LMT'):
        """
        Размещение bracket order (вход + стоп + тейк-профит)

        Args:
            action: 'BUY' или 'SELL'
            quantity: Количество контрактов
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            take_profit: Цена тейк-профита
            order_type: 'LMT' (лимитный) или 'MKT' (рыночный)
        """

        # Родительский ордер (вход)
        if order_type == 'LMT':
            parent = LimitOrder(action, quantity, entry_price)
        else:
            parent = MarketOrder(action, quantity)

        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False  # Не отправляем сразу

        # Стоп-лосс
        stop_loss_order = StopOrder(
            'SELL' if action == 'BUY' else 'BUY',
            quantity,
            stop_loss
        )
        stop_loss_order.orderId = self.ib.client.getReqId()
        stop_loss_order.parentId = parent.orderId
        stop_loss_order.transmit = False

        # Тейк-профит
        take_profit_order = LimitOrder(
            'SELL' if action == 'BUY' else 'BUY',
            quantity,
            take_profit
        )
        take_profit_order.orderId = self.ib.client.getReqId()
        take_profit_order.parentId = parent.orderId
        take_profit_order.transmit = True  # Отправляем все вместе

        # Размещаем все ордера
        trades = []
        for order in [parent, stop_loss_order, take_profit_order]:
            trade = self.ib.placeOrder(contract, order)
            trades.append(trade)

        self.active_orders[parent.orderId] = {
            'contract': contract,
            'trades': trades,
            'action': action,
            'entry': entry_price,
            'sl': stop_loss,
            'tp': take_profit
        }

        logging.info(f"📊 Bracket order placed: {action} {quantity} @ {entry_price}")
        logging.info(f"   SL: {stop_loss} | TP: {take_profit}")

        return trades

    def cancel_order(self, order_id):
        """Отмена ордера"""
        if order_id in self.active_orders:
            for trade in self.active_orders[order_id]['trades']:
                self.ib.cancelOrder(trade.order)
            del self.active_orders[order_id]
            logging.info(f"Order {order_id} cancelled")

    def get_account_balance(self):
        """Получение баланса счета"""
        account_values = self.ib.accountValues()
        for item in account_values:
            if item.tag == 'NetLiquidation' and item.currency == 'USD':
                return float(item.value)
        return 0.0

    def get_positions(self):
        """Получение текущих позиций"""
        positions = self.ib.positions()
        pos_dict = {}
        for pos in positions:
            pos_dict[pos.contract.symbol] = {
                'quantity': pos.position,
                'avgCost': pos.avgCost,
                'marketValue': pos.marketValue,
                'unrealizedPNL': pos.unrealizedPNL
            }
        return pos_dict

    def on_order_status(self, trade):
        """Обработчик изменения статуса ордера"""
        logging.info(f"Order Status: {trade.orderStatus.status} | {trade.contract.symbol}")

    def on_execution(self, trade, fill):
        """Обработчик исполнения ордера"""
        logging.info(f"✅ Order Filled: {fill.execution.side} {fill.execution.shares} @ {fill.execution.price}")


# ============================================================
# ИНТЕГРАЦИЯ С ВАШЕЙ СТРАТЕГИЕЙ
# ============================================================

class LiveInsideBarStrategy:
    """
    Живая торговля по стратегии Inside Bar с макро-фильтрами
    """

    def __init__(self, bot, symbol, risk_percent=1.0, risk_reward=2.75):
        self.bot = bot
        self.symbol = symbol
        self.risk_percent = risk_percent
        self.risk_reward = risk_reward
        self.atr_period = 14
        self.last_check_time = None

        # Создаем контракт
        if symbol == 'BRENTCMDUSD':
            self.contract = bot.create_contract('BRN', 'FUT', 'IPE', 'USD')
        elif symbol == 'XAUUSD':
            self.contract = bot.create_contract('XAUUSD', 'CASH')
        else:
            self.contract = bot.create_contract(symbol)

    def calculate_atr(self, df, period=14):
        """Расчет ATR (из вашего кода)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()
        return df

    def check_inside_bar_signal(self, df):
        """
        Проверка сигнала Inside Bar
        Returns: ('BUY'/'SELL'/None, entry_price, stop_loss)
        """
        if len(df) < 3:
            return None, None, None

        # Расчет ATR
        df = self.calculate_atr(df, self.atr_period)

        # Проверка Inside Bar
        df["Prev_High"] = df["High"].shift(1)
        df["Prev_Low"] = df["Low"].shift(1)
        df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

        # Проверяем предыдущую свечу (i-1)
        if not df.iloc[-2]["Is_Inside_Bar"]:
            return None, None, None

        # Inside Bar найден, смотрим на текущую свечу
        ib_high = df.iloc[-2]["High"]
        ib_low = df.iloc[-2]["Low"]
        curr_high = df.iloc[-1]["High"]
        curr_low = df.iloc[-1]["Low"]

        # Пробой вверх
        if curr_high > ib_high:
            return 'BUY', ib_high, ib_low
        # Пробой вниз
        elif curr_low < ib_low:
            return 'SELL', ib_low, ib_high

        return None, None, None

    def apply_macro_filter(self, direction, macro_score):
        """
        Применение макро-фильтра (из вашего кода)
        """
        if self.symbol == "BRENTCMDUSD" or self.symbol == "XAUUSD":
            if direction == "SELL" and macro_score <= 0:
                return True
            elif direction == "BUY" and macro_score >= 0:
                return True
            return False
        return True  # По умолчанию разрешаем

    def run_live(self, check_interval=300, macro_score=0):
        """
        Запуск живой торговли

        Args:
            check_interval: Интервал проверки в секундах (300 = 5 минут)
            macro_score: Текущий макро-счет (получайте из вашего MacroProcessor)
        """
        logging.info(f"🚀 Starting live strategy for {self.symbol}")
        logging.info(f"Risk: {self.risk_percent}% | R:R = {self.risk_reward}")

        try:
            while True:
                # 1. Получаем исторические данные
                df = self.bot.get_historical_data(self.contract, '5 D', '4 hours')

                if df.empty:
                    logging.warning("No data received, retrying...")
                    time.sleep(60)
                    continue

                # 2. Проверяем сигнал
                direction, entry_price, stop_loss = self.check_inside_bar_signal(df)

                if direction:
                    # 3. Применяем макро-фильтр
                    if not self.apply_macro_filter(direction, macro_score):
                        logging.info(f"⚠️ Signal {direction} blocked by macro filter (score={macro_score})")
                        time.sleep(check_interval)
                        continue

                    # 4. Рассчитываем позицию
                    account_balance = self.bot.get_account_balance()
                    quantity = self.bot.calculate_position_size(
                        entry_price, stop_loss, self.risk_percent, account_balance
                    )

                    if quantity == 0:
                        logging.warning("Position size = 0, skipping trade")
                        time.sleep(check_interval)
                        continue

                    # 5. Рассчитываем тейк-профит
                    risk_distance = abs(entry_price - stop_loss)
                    if direction == 'BUY':
                        take_profit = entry_price + (risk_distance * self.risk_reward)
                    else:
                        take_profit = entry_price - (risk_distance * self.risk_reward)

                    # 6. Размещаем ордер
                    logging.info(f"🎯 SIGNAL DETECTED: {direction}")
                    logging.info(f"   Macro Score: {macro_score}")
                    logging.info(f"   Account Balance: ${account_balance:,.2f}")

                    self.bot.place_bracket_order(
                        self.contract,
                        direction,
                        quantity,
                        entry_price,
                        stop_loss,
                        take_profit,
                        order_type='LMT'  # Используем лимитный ордер на пробой
                    )

                    # Ждем закрытия сделки перед новой
                    time.sleep(3600)  # 1 час

                else:
                    logging.info(f"No signal. Waiting {check_interval}s...")

                # Ждем следующей проверки
                time.sleep(check_interval)

        except KeyboardInterrupt:
            logging.info("⏹️ Strategy stopped by user")
        except Exception as e:
            logging.error(f"❌ Error in strategy: {e}")
        finally:
            self.bot.disconnect()


# ============================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================

def main():
    """
    Запуск live trading бота
    """

    # 1. Подключаемся к IBKR
    bot = IBKRTradingBot(
        host='127.0.0.1',
        port=7497,  # Paper trading
        client_id=1
    )

    if not bot.connect():
        print("Failed to connect to IBKR")
        return

    # 2. Проверяем баланс
    balance = bot.get_account_balance()
    print(f"\n💰 Account Balance: ${balance:,.2f}\n")

    # 3. Получаем макро-данные (используйте ваш MacroProcessor)
    # Пример: для демо используем нейтральное значение
    from datetime import datetime

    # Здесь вы можете интегрировать ваш MacroProcessor
    # macro_proc = MacroProcessor(FRED_API_KEY)
    # df_macro = macro_proc.fetch_and_process()
    # current_macro_score = df_macro.iloc[-1]['USD_Score']

    current_macro_score = 0  # Для демо

    # 4. Запускаем стратегию
    strategy = LiveInsideBarStrategy(
        bot=bot,
        symbol='CL',  # или 'BRENTCMDUSD'
        risk_percent=1.0,
        risk_reward=2.75
    )

    strategy.run_live(
        check_interval=300,  # Проверка каждые 5 минут
        macro_score=current_macro_score
    )


if __name__ == "__main__":
    main()