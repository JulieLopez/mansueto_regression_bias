"""
Cook County check: how many sales are there each year in the First American
data, and is 2025 really empty for residential?

Run from Terminal:
    python3 check_cook_years.py

Outputs (saved in the same folder):
    cook_sales_by_year.csv   counts by year: all classes + residential
    cook_sales_by_year.png   bar chart of residential sales by year
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save the chart without opening a window
import matplotlib.pyplot as plt

# ---- Settings ---------------------------------------------------------------
DATA_DIR = (Path.home() / "regression_bias/first_american_data"
            / "First American Data Joined")
COOK = DATA_DIR / "17031.parquet"
CHECK_YEAR = 2025

# ---- 1. Load Cook ------------------------------------------------------------
# Loading Cook's own file directly (not the whole national folder)
print(f"Reading {COOK}\n")
cook = pd.read_parquet(COOK, columns=["FIPS", "Year", "PropertyClassID",
                                      "MarketTotalValue", "SaleAmt"])

# Sanity check: is every row in this file actually Cook?
print("FIPS codes in this file:", cook["FIPS"].unique().tolist())
print(f"Total rows: {len(cook):,}")
print(f"Rows with a missing Year: {cook['Year'].isna().sum():,}")
print(f"Years covered: {cook['Year'].min()} to {cook['Year'].max()}\n")

# ---- 2. Counts by year -------------------------------------------------------
res = cook[cook["PropertyClassID"] == "R"]

by_year = pd.DataFrame({
    "all_classes": cook.groupby("Year").size(),
    "residential": res.groupby("Year").size(),
    # residential rows the simulation could actually use
    # (has a market value and a real sale price)
    "residential_usable": res[res["MarketTotalValue"].notna()
                              & (res["SaleAmt"] > 100)
                              & (res["MarketTotalValue"] > 100)]
                          .groupby("Year").size(),
})

# Make sure CHECK_YEAR shows up as a row even if it has zero sales,
# so the table itself shows the zero
all_years = range(int(cook["Year"].min()), max(int(cook["Year"].max()), CHECK_YEAR) + 1)
by_year = by_year.reindex(all_years, fill_value=0).fillna(0).astype(int)
by_year.index.name = "Year"

print("Cook County sales by year:")
print(by_year.to_string())

# ---- 3. The proof for CHECK_YEAR --------------------------------------------
n_all = (cook["Year"] == CHECK_YEAR).sum()
n_res = (res["Year"] == CHECK_YEAR).sum()

print(f"\n---- {CHECK_YEAR} check ----")
print(f"Rows in {CHECK_YEAR}, any property class: {n_all:,}")
print(f"Rows in {CHECK_YEAR}, residential (R):     {n_res:,}")
if n_res == 0:
    print(f"CONFIRMED: no residential sales for Cook in {CHECK_YEAR} "
          f"in this dataset. Latest year is {int(res['Year'].max())}.")
else:
    print(f"Cook DOES have {n_res:,} residential sales in {CHECK_YEAR}.")

# ---- 4. Save the table and a chart -------------------------------------------
by_year.to_csv("cook_sales_by_year.csv")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(by_year.index, by_year["residential"], color="steelblue")
ax.set_title("Cook County residential sales by year (First American data)")
ax.set_xlabel("Year")
ax.set_ylabel("Number of sales")
ax.set_xticks(list(by_year.index))
ax.tick_params(axis="x", rotation=45)
for x, y in zip(by_year.index, by_year["residential"]):
    ax.text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig("cook_sales_by_year.png", dpi=120)

print("\nSaved cook_sales_by_year.csv and cook_sales_by_year.png")