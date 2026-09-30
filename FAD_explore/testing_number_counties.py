# ignore

from pathlib import Path
import pyarrow.dataset as ds

DATA_DIR = (Path.home() / "regression_bias/first_american_data"
            / "First American Data Joined")

# 1. How many county files are there?
files = list(DATA_DIR.glob("*.parquet"))
print(f"{len(files)} county files")

# 2. Count residential sales for every county and year
dataset = ds.dataset(DATA_DIR, format="parquet")
t = dataset.to_table(columns=["FIPS", "Year"],
                     filter=ds.field("PropertyClassID") == "R")
counts = (t.group_by(["FIPS", "Year"])
           .aggregate([("Year", "count")])
           .to_pandas()
           .rename(columns={"Year_count": "n_sales"}))

print(f"{counts['FIPS'].nunique()} counties with residential sales")
print(f"{counts['FIPS'].str[:2].nunique()} states (incl. DC/territories)\n")

# 3. By year: how many counties have data, and how many have 100+ sales
by_year = counts.groupby("Year").agg(
    counties=("FIPS", "nunique"),
    counties_100plus=("n_sales", lambda s: (s >= 100).sum()),
    total_sales=("n_sales", "sum"),
)
print(by_year.to_string())

# Save the full county-by-year table to look at in Excel
counts.sort_values(["FIPS", "Year"]).to_csv("county_year_counts.csv", index=False)
print("\nSaved county_year_counts.csv")