import pandas as pd
from sqlalchemy.sql.sqltypes import NULLTYPE

df = pd.read_csv("/Users/maxxxxx/PycharmProjects/InsideBarStrg/inside_bar_rub/macro_data/Calender_data.csv")
df["date"] = pd.to_datetime(df["date"],format="%d/%m/%Y")
us_data = df[(df["zone"] == "united states") & ((df["importance"] == "medium") | (df["importance"] == "high")) ]
japan_data = df[(df["zone"] == "japan") & ((df["importance"] == "medium") | (df["importance"] == "high"))]

print(us_data.head())