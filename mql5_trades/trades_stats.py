import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("5year_4h_jpy.csv")

df.plot(x="OpenTime", y="Balance", figsize=(15, 5))
plt.show()

print(df[["Balance"]].min())
