import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/julie/regression_bias/all_counties_regression_bias/first_american_data")
r = pd.read_csv("results_2024_1sample/county_results_2024_1sample.csv", dtype={"fips": str})
missing = r[r["status"] == "no residential sales in 2024"]["fips"]

reasons = []
for fips in missing:
    path = next(DATA_DIR.rglob(f"{fips}.parquet"))
    df = pd.read_parquet(path, columns=["Year", "PropertyClassID", "MarketTotalValue"])
    d24 = df[(df["Year"] == 2024) & (df["PropertyClassID"] == "R")]
    if len(d24) == 0:
        reasons.append("no 2024 data")
    elif d24["MarketTotalValue"].isna().all():
        reasons.append("2024 data, but no market value")
    else:
        reasons.append("other")

print(pd.Series(reasons).value_counts())

states = []
for fips, reason in zip(missing, reasons):
    if reason == "2024 data, but no market value":
        path = next(DATA_DIR.rglob(f"{fips}.parquet"))
        s = pd.read_parquet(path, columns=["SitusState"])["SitusState"].mode()
        states.append(s.iat[0] if len(s) else "?")
print(pd.Series(states).value_counts())