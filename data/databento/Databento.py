import databento as db

client = db.Historical("API-KEY")


common = dict(dataset="GLBX.MDP3", symbols=["CL.v.0"], stype_in="continuous", end="2026-07-02")

START = "2023-07-01"
#
print(client.metadata.get_cost(schema="trades", start=START, **common))

client.timeseries.get_range(
    schema="trades", start=START, **common,
    path="cl_trades_2023_2026.dbn.zst",
)
print("done")