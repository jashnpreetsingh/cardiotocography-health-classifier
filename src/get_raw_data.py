from ucimlrepo import fetch_ucirepo

ctg = fetch_ucirepo(id=193)

X = ctg.data.features
y = ctg.data.targets

print(X.shape)       # expect (2126, 21)
print(y.columns)  

import pandas as pd

df = X.copy()
df["NSP"] = y["NSP"]

df.to_csv("../data/ctg_raw.csv", index=False)
df.head()