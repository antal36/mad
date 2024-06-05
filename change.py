import pandas as pd
df = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/heart_attack1/Medicaldataset.csv")
df["Result"] = df["Result"].map({"positive":1, "negative":0})
df.to_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/heart_attack1/Medicaldataset_final.csv")