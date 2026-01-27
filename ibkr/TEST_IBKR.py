"""
ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ
Полностью готовая к запуску стратегия с исправленными контрактами
Добавьте это в конец вашего основного файла
"""

from ib_insync import *
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class FixedIBKRBot:
    """Исправленная версия торгового бота"""

    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self):
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logging.info(f"✅ Connected to IBKR on port {self.port}")
            return True
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            return False

    def create_contract(self, symbol):
        """Создание контракта с правильными параметрами"""

        # НЕФТЬ
        if symbol in ['BRENTCMDUSD', 'BRENT', 'BRN']:
            contract = ContFuture('BZ', exchange='NYMEX')  # Brent на NYMEX, не IPE!
            desc = "Brent Crude (NYMEX: BZ)"

        elif symbol in ['CL', 'WTI']:
            contract = ContFuture('CL', exchange='NYMEX')
            desc = "WTI Crude (NYMEX: CL)"

        # ЗОЛОТО
        elif symbol in ['XAUUSD', 'GOLD']:
            contract = Forex('XAUUSD')
            desc = "Gold Forex (XAUUSD)"

        elif symbol == 'GC':
            contract = ContFuture('GC', exchange='COMEX')
            desc = "Gold Futures (COMEX: GC)"

        # КРИПТО
        elif symbol in ['BTC', 'BTCUSD', 'BTCUSD1']:
            contract = ContFuture('MBT', exchange='CME')
            desc = "Micro Bitcoin (CME: MBT)"

        # FOREX
        elif symbol == 'EURUSD':
            contract = Forex('EURUSD')
            desc = "EUR/USD"

        # АКЦИИ
        elif len(symbol) <= 5 and symbol.isupper():
            contract = Stock(symbol, 'SMART', 'USD')
            desc = f"Stock ({symbol})"

        else:
            logging.error(f"❌ Unknown symbol: {symbol}")
            return None

        # Квалификация
        try:
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                logging.info(f"✅ {desc} - Contract qualified")
                return qualified[0]
            else:
                logging.error(f"❌ {desc} - Failed to qualify")
                return None
        except Exception as e:
            logging.error(f"❌ Error: {e}")
            return None

    def get_historical_data(self, contract, duration='5 D', bar_size='4 hours'):
        """Получение исторических данных"""
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False
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
            return pd.DataFrame()

        except Exception as e:
            logging.error(f"❌ Historical data error: {e}")
            return pd.DataFrame()

    def get_account_balance(self):
        """Получение баланса"""
        try:
            account_values = self.ib.accountValues()
            for item in account_values:
                if item.tag == 'NetLiquidation' and item.currency == 'USD':
                    return float(item.value)
            return 10000.0  # Default для paper trading
        except:
            return 10000.0

    def place_bracket_order(self, contract, action, quantity, entry_price, stop_loss, take_profit):
        """Размещение bracket order"""
        try:
            parent = LimitOrder(action, quantity, entry_price)
            parent.orderId = self.ib.client.getReqId()
            parent.transmit = False

            stop = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss)
            stop.orderId = self.ib.client.getReqId()
            stop.parentId = parent.orderId
            stop.transmit = False

            profit = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit)
            profit.orderId = self.ib.client.getReqId()
            profit.parentId = parent.orderId
            profit.transmit = True

            trades = []
            for order in [parent, stop, profit]:
                trade = self.ib.placeOrder(contract, order)
                trades.append(trade)

            logging.info(f"📊 Order placed: {action} {quantity} @ {entry_price}")
            logging.info(f"   SL: {stop_loss} | TP: {take_profit}")
            return trades

        except Exception as e:
            logging.error(f"❌ Order error: {e}")
            return None

    def disconnect(self):
        self.ib.disconnect()


class SimpleInsideBarStrategy:
    """Упрощенная версия стратегии для тестирования"""

    def __init__(self, bot, symbol, risk_percent=1.0, risk_reward=2.75):
        self.bot = bot
        self.symbol = symbol
        self.risk_percent = risk_percent
        self.risk_reward = risk_reward

        # Создаем контракт
        self.contract = bot.create_contract(symbol)
        if not self.contract:
            raise ValueError(f"Failed to create contract for {symbol}")

    def calculate_atr(self, df, period=14):
        """Расчет ATR"""
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

    def check_signal(self, df):
        """Проверка Inside Bar сигнала"""
        if len(df) < 3:
            return None, None, None

        df = self.calculate_atr(df)

        # Inside Bar логика
        df["Prev_High"] = df["High"].shift(1)
        df["Prev_Low"] = df["Low"].shift(1)
        df["Is_Inside_Bar"] = (df["High"] < df["Prev_High"]) & (df["Low"] > df["Prev_Low"])

        if not df.iloc[-2]["Is_Inside_Bar"]:
            return None, None, None

        ib_high = df.iloc[-2]["High"]
        ib_low = df.iloc[-2]["Low"]
        curr_high = df.iloc[-1]["High"]
        curr_low = df.iloc[-1]["Low"]

        if curr_high > ib_high:
            return 'BUY', ib_high, ib_low
        elif curr_low < ib_low:
            return 'SELL', ib_low, ib_high

        return None, None, None

    def run_once(self):
        """Одна проверка сигнала (для тестирования)"""
        logging.info(f"🔍 Checking signal for {self.symbol}")

        # Получаем данные
        df = self.bot.get_historical_data(self.contract, '5 D', '4 hours')

        if df.empty:
            logging.warning("No data received")
            return False

        logging.info(f"📊 Received {len(df)} bars")
        logging.info(f"   Last bar: {df.iloc[-1]['DateTime']} | C: {df.iloc[-1]['Close']}")

        # Проверяем сигнал
        direction, entry, stop = self.check_signal(df)

        if direction:
            logging.info(f"🎯 SIGNAL: {direction}")
            logging.info(f"   Entry: {entry} | Stop: {stop}")

            # Расчет позиции
            balance = self.bot.get_account_balance()
            risk_distance = abs(entry - stop)
            position_size = int((balance * self.risk_percent / 100) / risk_distance)

            # Расчет TP
            if direction == 'BUY':
                take_profit = entry + (risk_distance * self.risk_reward)
            else:
                take_profit = entry - (risk_distance * self.risk_reward)

            logging.info(f"   Position: {position_size} | TP: {take_profit}")

            # Размещение ордера (раскомментируйте для реальной торговли)
            # self.bot.place_bracket_order(
            #     self.contract, direction, position_size,
            #     entry, stop, take_profit
            # )

            return True
        else:
            logging.info("No signal")
            return False


# ============================================================
# ГОТОВЫЙ ПРИМЕР ЗАПУСКА
# ============================================================

def run_strategy_test():
    """
    Готовый к запуску пример
    """

    print("\n" + "=" * 70)
    print("🚀 STARTING INSIDE BAR STRATEGY")
    print("=" * 70)

    # 1. Подключение
    bot = FixedIBKRBot(port=7497)
    if not bot.connect():
        return

    # 2. Проверка баланса
    balance = bot.get_account_balance()
    print(f"\n💰 Account Balance: ${balance:,.2f}")

    # 3. Выбор символа
    # Попробуйте эти символы по порядку:
    test_symbols = ['XAUUSD', 'CL', 'EURUSD', 'AAPL']

    working_symbol = None
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        contract = bot.create_contract(symbol)
        if contract:
            working_symbol = symbol
            print(f"✅ {symbol} is working!")
            break

    if not working_symbol:
        print("\n❌ No working symbols found. Check your Paper Trading setup.")
        bot.disconnect()
        return

    # 4. Запуск стратегии
    try:
        strategy = SimpleInsideBarStrategy(
            bot=bot,
            symbol=working_symbol,
            risk_percent=1.0,
            risk_reward=2.75
        )

        print(f"\n✅ Strategy initialized for {working_symbol}")
        print("Running signal check...")

        strategy.run_once()

        print("\n" + "=" * 70)
        print("✅ TEST COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. If you see a signal → strategy is working!")
        print("2. Uncomment order placement in run_once()")
        print("3. Add loop for continuous monitoring")

    except Exception as e:
        logging.error(f"❌ Strategy error: {e}")

    finally:
        bot.disconnect()


if __name__ == "__main__":
    run_strategy_test()