import yfinance as yf
import pandas as pd


"""Parsing Yen Volume Data from Yahoo Finance:"""
YEN_1h = yf.download(tickers=["6J=F"], period="1y", interval="1h")
YEN_1h.columns=[col[0]for col in YEN_1h.columns]
YEN_4h = yf.download(tickers=["6J=F"], period="1y", interval="4h")
YEN_4h.columns=[col[0]for col in YEN_4h.columns]
YEN_D = yf.download(tickers=["6J=F"], period="1y", interval="1d")
YEN_D.columns=[col[0]for col in YEN_D.columns]

"""Parsing Gold Volume Data from Yahoo Finance:"""
GOLD_1h = yf.download(tickers=["GC=F"], period="1y", interval="1h")
GOLD_1h.columns=[col[0]for col in GOLD_1h.columns]
GOLD_4h = yf.download(tickers=["GC=F"], period="1y", interval="4h")
GOLD_4h.columns=[col[0]for col in GOLD_4h.columns]
GOLD_D = yf.download(tickers=["GC=F"], period="1y", interval="1d")
GOLD_D.columns=[col[0]for col in GOLD_D.columns]

"""Parsing Oil Volume Data from Yahoo Finance:"""
OIL_1h = yf.download(tickers=["BZ=F"], period="1y", interval="1h")
OIL_1h.columns=[col[0]for col in OIL_1h.columns]
OIL_4h = yf.download(tickers=["BZ=F"], period="1y", interval="4h")
OIL_1h.columns=[col[0]for col in OIL_1h.columns]
OIL_D = yf.download(tickers=["BZ=F"], period="1y", interval="1d")
OIL_1h.columns=[col[0]for col in OIL_1h.columns]

print(OIL_1h.head())