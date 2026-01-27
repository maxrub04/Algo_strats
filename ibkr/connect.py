from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

base_contract = Future(symbol='CL', exchange='NYMEX', currency='USD')

print("Ищем активный контракт...")
details = ib.reqContractDetails(base_contract)

if not details:
    print("Контракты не найдены! Проверьте тикер/биржу.")
else:
    active_contract_details = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)[0]
    contract = active_contract_details.contract

    print(f" Найден активный контракт: {contract.localSymbol}")
    print(f"   Месяц: {contract.lastTradeDateOrContractMonth}")
    print(f"   Экспирация: {active_contract_details.realExpirationDate}")

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 hour',
        whatToShow='TRADES',
        useRTH=True
    )

    df = util.df(bars)
    print(df)

ib.disconnect()