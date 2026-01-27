import pandas as pd
import numpy as np
from fredapi import Fred
from ib_insync import *
import asyncio
import logging
from datetime import datetime

# --- CONFIGURATION ---
FRED_API_KEY = ''
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # 7497 = Paper Trading, 7496 = Live Trading
CLIENT_ID = 1

# STRATEGY SETTINGS
SYMBOL_IB = 'CL'  # Brent Crude Oil Futures (NYMEX)
EXCHANGE = 'NYMEX'  # Биржа
CURRENCY = 'USD'
TIMEFRAME = '4 hours'  # Формат IBKR
RISK_REWARD = 2.75
RISK_PERCENT = 1.0  # 1% риск от депозита

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


# --- PART 1: MACRO PROCESSOR (ADAPTED FOR LIVE) ---
class MacroProcessorLive:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.indicators = {
            'PAYEMS': 'NFP_Total_Jobs',
            'DGS2': 'US_Treasury_2Y_Yield',
            'DFII10': 'TIPS_10Y_Real_Yield',
            'IMPGS': 'Imports',
            'EXPGS': 'Exports',
            'BOPGSTB': 'Trade_Balance',
        }
        self.current_score = 0

    def update_score(self):
        """Fetches latest data and calculates USD Score"""
        logger.info("Updating Macro Score...")
        try:
            # Fetch only the last few points to calculate diffs
            data = {}
            for ticker, name in self.indicators.items():
                try:
                    # Fetch last 3 months to be safe for diff calculation
                    series = self.fred.get_series(ticker, observation_start=pd.Timestamp.now() - pd.Timedelta(days=90))
                    data[name] = series.iloc[-1]
                    data[f"{name}_prev"] = series.iloc[-2]
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")

            if not data:
                return 0

            score = 0

            # Rule 1: NFP Trend
            if data.get('NFP_Total_Jobs', 0) > data.get('NFP_Total_Jobs_prev', 0):
                score += 1
            else:
                score -= 1

            # Rule 3: TIPS
            tips_diff = data.get('TIPS_10Y_Real_Yield', 0) - data.get('TIPS_10Y_Real_Yield_prev', 0)
            if tips_diff > 0:
                score += 1
            else:
                score -= 1

            # Rule 4: Inflation vs Deflation (Exports/Imports)
            imp_diff = data.get('Imports', 0) - data.get('Imports_prev', 0)
            exp_diff = data.get('Exports', 0) - data.get('Exports_prev', 0)

            if imp_diff < 0 and exp_diff > 0:
                score += 1
            elif imp_diff > 0 and exp_diff < 0:
                score -= 1

            # Rule 5: Trade Balance
            tb_diff = data.get('Trade_Balance', 0) - data.get('Trade_Balance_prev', 0)
            if tb_diff > 0:
                score += 1
            else:
                score -= 1

            self.current_score = score
            logger.info(f"Updated USD Score: {self.current_score}")
            return score

        except Exception as e:
            logger.error(f"Macro update failed: {e}")
            return 0


# --- PART 2: TRADING BOT ---
class InsideBarBot:
    def __init__(self):
        self.ib = IB()
        self.macro = MacroProcessorLive(FRED_API_KEY)
        self.contract = None
        self.bars = []
        self.in_position = False
        self.account_value = 0.0

    async def run(self):
        # 1. Connect
        logger.info("Connecting to IBKR...")
        try:
            await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        except Exception as e:
            logger.error(f"Connection failed: {e}. Check TWS settings.")
            return

        # 2. Setup Contract (Futures for Brent)
        # Важно: Для фьючерсов нужно указать дату экспирации.
        # ContFuture автоматически склеивает контракты, но торговать им напрямую нельзя.
        # Для торговли мы найдем активный фьючерс.
        self.contract = Future(symbol='BZ', lastTradeDateOrContractMonth='202503', exchange=EXCHANGE, currency=CURRENCY)
        # Уточняем контракт (получаем ID и детали)
        details = await self.ib.reqContractDetailsAsync(self.contract)
        if not details:
            logger.error("Contract not found!")
            return
        self.contract = details[0].contract
        logger.info(f"Trading Contract: {self.contract.localSymbol}")

        # 3. Initial Macro Update
        self.macro.update_score()

        # 4. Request Historical Data & Keep Up to Date
        # Мы запрашиваем данные и подписываемся на обновления
        self.bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime='',
            durationStr='10 D',
            barSizeSetting=TIMEFRAME,
            whatToShow='TRADES',
            useRTH=False,  # Regular Trading Hours = False (торгуем 24/7)
            formatDate=1,
            keepUpToDate=True  # Критично: получать новые бары в реальном времени
        )

        # Привязываем функцию обработки к событию обновления баров
        self.bars.updateEvent += self.on_bar_update

        # 5. Keep alive
        logger.info("Bot started. Waiting for bars...")
        while self.ib.isConnected():
            await asyncio.sleep(1)
            # Раз в час обновляем макро
            if datetime.now().minute == 0 and datetime.now().second < 5:
                self.macro.update_score()

    def on_bar_update(self, bars, has_new_bar):
        """
        Triggered when a bar is updated or a new bar is created.
        """
        if not has_new_bar:
            return  # Ждем только закрытия свечи

        # Получаем DataFrame из последних свечей
        df = util.df(bars)

        # Нам нужно как минимум 3 свечи: [Previous, InsideBar, Current_Forming]
        # bars[-1] - это формирующаяся свеча.
        # bars[-2] - это только что закрытая свеча (Potential Inside Bar)
        # bars[-3] - это свеча до нее (Mother Bar)

        if len(df) < 3:
            return

        # Определяем индексы
        prev_bar = df.iloc[-3]  # Mother Bar
        curr_bar = df.iloc[-2]  # Inside Bar (Just closed)

        logger.info(f"New Bar Closed: {curr_bar['date']} | H:{curr_bar['high']} L:{curr_bar['low']}")

        # 1. Check Inside Bar Pattern
        is_inside = (curr_bar['high'] < prev_bar['high']) and (curr_bar['low'] > prev_bar['low'])

        if is_inside:
            logger.info(">>> INSIDE BAR DETECTED <<<")
            self.process_signal(curr_bar)

    def process_signal(self, ib_bar):
        # 2. Check Macro Filter
        score = self.macro.current_score
        # Логика из вашего кода:
        # BRENT (Oil) часто коррелирует с USD.
        # Если Score > 0 (Strong USD), Нефть может падать -> Разрешаем SELL.
        # Если Score < 0 (Weak USD), Нефть может расти -> Разрешаем BUY.

        # Определяем уровни
        buy_trigger = ib_bar['high']
        sell_trigger = ib_bar['low']

        # Риск дистанция
        risk_dist = ib_bar['high'] - ib_bar['low']
        if risk_dist == 0: return

        # Проверяем позиции (чтобы не дублировать)
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.conId == self.contract.conId:
                logger.info("Already in position. Skipping.")
                return

        # --- LOGIC FOR BUY ---
        # Если доллар слабый или нейтральный (Score <= 0), разрешаем покупку
        if score <= 0:
            self.place_bracket_order('BUY', buy_trigger, ib_bar['low'], risk_dist)

        # --- LOGIC FOR SELL ---
        # Если доллар сильный или нейтральный (Score >= 0), разрешаем продажу
        if score >= 0:
            self.place_bracket_order('SELL', sell_trigger, ib_bar['high'], risk_dist)

    def place_bracket_order(self, action, entry_price, stop_loss_price, risk_dist):
        # 1. Calculate Quantity
        account = [v for v in self.ib.accountSummary() if v.tag == 'NetLiquidation'][0]
        balance = float(account.value)

        risk_money = balance * (RISK_PERCENT / 100.0)
        # Для фьючерсов нужно учитывать Multiplier (у BZ он 1000 баррелей)
        # Если CFD, то мультипликатор обычно 1.
        multiplier = int(self.contract.multiplier) if self.contract.multiplier else 1

        # Formula: (Risk $) / (Risk Dist * Multiplier)
        qty = risk_money / (risk_dist * multiplier)
        qty = round(qty)
        if qty < 1: qty = 1  # Минимальный лот

        # Take Profit Price
        if action == 'BUY':
            tp_price = entry_price + (risk_dist * RISK_REWARD)
            reverse_action = 'SELL'
        else:
            tp_price = entry_price - (risk_dist * RISK_REWARD)
            reverse_action = 'BUY'

        logger.info(
            f"Placing {action} STOP Order. Entry: {entry_price}, SL: {stop_loss_price}, TP: {tp_price}, Qty: {qty}")

        # --- BRACKET ORDER CONSTRUCTION ---

        # 1. Parent Order (Stop Entry) - Вход на пробой
        # Используем STOP ордер, чтобы войти, когда цена пробьет уровень Inside Bar
        parent = Order()
        parent.orderId = self.ib.client.getReqId()
        parent.action = action
        parent.orderType = 'STP'  # Stop Order (Pending)
        parent.auxPrice = entry_price  # Цена активации
        parent.totalQuantity = qty
        parent.transmit = False  # Не отправлять, пока не прикрепим SL/TP

        # 2. Take Profit (Child)
        tp_order = Order()
        tp_order.orderId = self.ib.client.getReqId()
        tp_order.action = reverse_action
        tp_order.orderType = 'LMT'
        tp_order.lmtPrice = tp_price
        tp_order.totalQuantity = qty
        tp_order.parentId = parent.orderId
        tp_order.transmit = False

        # 3. Stop Loss (Child)
        sl_order = Order()
        sl_order.orderId = self.ib.client.getReqId()
        sl_order.action = reverse_action
        sl_order.orderType = 'STP'
        sl_order.auxPrice = stop_loss_price
        sl_order.totalQuantity = qty
        sl_order.parentId = parent.orderId
        sl_order.transmit = True  # ОТПРАВИТЬ ВСЮ ПАЧКУ

        trades = self.ib.placeOrder(self.contract, parent)
        self.ib.placeOrder(self.contract, tp_order)
        self.ib.placeOrder(self.contract, sl_order)

        logger.info(f"Orders placed! ID: {parent.orderId}")


# --- EXECUTION ---
if __name__ == '__main__':
    bot = InsideBarBot()
    try:
        asyncio.run(bot.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")