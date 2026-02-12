import MetaTrader5 as mt5
import pandas as pd
import config_test as config
import time

SYMBOL = config.SYMBOL
VOLUME = config.VOLUME
MAGIC = config.MAGIC
ORDER_TYPE = mt5.ORDER_TYPE_BUY
STOP_LOSS = config.STOP_LOSS
TAKE_PROFIT = config.TAKE_PROFIT
PRICE_DEVIATION = config.PRICE_DEVIATION

LOGIN = #your login
PASSWORD = #your password
SERVER = #server name

def main():
    if not login():
        print("login failed, error code =", mt5.last_error())
        return

    if Init():
        while Loop():
            time.sleep(5)
    DeInit()

    return

def login():
    if not mt5.initialize():
        print("initialize failed, error code =", mt5.last_error())
        return False

    authorized = mt5.login(
        login = LOGIN,
        password = PASSWORD,
        server = SERVER)

    if not authorized:
        print("failed to connect at account #{}, error code: {}".format(
            LOGIN, mt5.last_error()))
        return False

    account = mt5.account_info()
    print(f"Connected to acc: {account.login}")
    print(f"Balance: {account.balance}")
    return True

def get_ohlc_data():
    bars= mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 10)
    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def Init():

    if not mt5.symbol_select(SYMBOL, True):
        print("symbol_select failed, error code =", mt5.last_error())
        return False
    return True


def Loop():
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is not None:
        print(f"Min Vol: {symbol_info.volume_min}")
        print(f"Masx Vol: {symbol_info.volume_max}")
        print(f"Step vOL: {symbol_info.volume_step}")
        print(f"Min SL: {symbol_info.trade_stops_level}")
        print(f"Тick size: {symbol_info.trade_tick_size}")
        print(f"Tick value: {symbol_info.trade_tick_value}")

        min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
        print(f"min distance to the stop: {min_stop_distance}")


    ohlc_data = get_ohlc_data()
    if ohlc_data is not None:
        print("\nLast 3 bars")
        print(ohlc_data[['time', 'open', 'high', 'low', 'close']].tail(3))

    price_info = mt5.symbol_info_tick(SYMBOL)
    if price_info is None:
        print("copy_tick_price failed, error code =", mt5.last_error())
        return True

    if ORDER_TYPE == mt5.ORDER_TYPE_BUY:
        price = price_info.ask
        stop_loss = price_info.bid - STOP_LOSS
        take_profit = price_info.bid + TAKE_PROFIT
    else:
        price = price_info.bid
        stop_loss = price_info.ask + STOP_LOSS
        take_profit = price_info.ask - TAKE_PROFIT

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "magic": MAGIC,
        "volume": float(VOLUME),
        "type": ORDER_TYPE,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": PRICE_DEVIATION,
        "type_filling": mt5.ORDER_FILLING_FOK,

    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("oder_send failed, error code =", result.retcode)
        print(result)
    else:
        print("oder_send done")
        print(result)

    return True

def DeInit():
    mt5.shutdown()
    return

if __name__ == "__main__":
    main()


