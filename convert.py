import pandas as pd

df = pd.read_csv("aita_cleaned.csv")
df.to_csv("aita_cleaned.csv.gz", index=False, compression="gzip")

