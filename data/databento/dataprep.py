import databento as db

df = db.DBNStore.from_file("cl_trades_2023_2026.dbn.zst").to_df()
print(df.shape, df.index.min(), df.index.max())
print(df.head())