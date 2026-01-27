import pandas as pd
from ib_insync import *
import os

# --- НАСТРОЙКИ ---
HOST = '127.0.0.1'
PORT = 7497
CLIENT_ID = 102  # Новый ID
SYMBOL = 'CL'
EXCHANGE = 'NYMEX'
TIMEFRAME = '4 hours'
DURATION = '8 Y'  # <-- ЗАПРАШИВАЕМ ВСЁ СРАЗУ (Попробуйте '5 Y', если пройдет - '10 Y')

OUTPUT_FOLDER = "data_export"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def download_one_shot():
    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID)
        print("✅ Connected to IBKR")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    # Создаем Continuous Future
    contract = ContFuture(symbol=SYMBOL, exchange=EXCHANGE, currency='USD')
    print(f"🎯 Target Contract: {SYMBOL} (Continuous)")

    print(f"🚀 Скачиваем {DURATION} истории одним запросом...")

    try:
        # ВАЖНО: endDateTime должен быть ПУСТЫМ ('')
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',  # <--- ПУСТО (Требование API для ContFuture)
            durationStr=DURATION,  # <--- Весь период сразу
            barSizeSetting=TIMEFRAME,
            whatToShow='TRADES',  # Если будет ошибка "No data", замените на 'MIDPOINT'
            useRTH=False,
            formatDate=1,
            timeout=120  # Даем серверу подумать подольше
        )
    except Exception as e:
        print(f"❌ Ошибка API: {e}")
        bars = []

    if bars:
        df = util.df(bars)
        df.set_index('date', inplace=True)

        filename = f"{OUTPUT_FOLDER}/{SYMBOL}_{TIMEFRAME.replace(' ', '')}_FULL.csv"
        df.to_csv(filename)

        print(f"\n🎉 УСПЕХ! Скачано {len(df)} свечей.")
        print(f"📅 Период: {df.index[0]} — {df.index[-1]}")
        print(f"💾 Файл: {filename}")
    else:
        print("⚠️ Данные не получены. Возможно, период слишком большой для одного запроса.")

    ib.disconnect()


if __name__ == "__main__":
    download_one_shot()