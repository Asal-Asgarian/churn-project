import pandas as pd

df = pd.read_csv("churn_data.csv")

print("First rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nCompare customers:\n")
print(df.groupby("churn")[["tenure", "support_calls"]].mean())

print("\nChurn distribution:")
print(df["churn"].value_counts(normalize=True))