import os
import pyarrow.dataset as ds
import pandas as pd

path = os.path.expanduser("~/regression_bias/first_american_data")

for root, dirs, files in os.walk(path):
    for f in files[:20]:
        print(os.path.join(root, f))

dataset = ds.dataset(path, format="parquet", partitioning="hive")

print(dataset.schema)        # every column name + type
print(dataset.count_rows())  # total rows, without loading data

samp = dataset.head(5000).to_pandas()

dict_df = pd.DataFrame({
    "column": dataset.schema.names,
    "type": [str(f.type) for f in dataset.schema],
    "example_values": [
        " | ".join(map(str, samp[c].dropna().unique()[:3]))
        for c in dataset.schema.names
    ],
    "pct_missing_in_sample": [
        round(samp[c].isna().mean() * 100, 1) for c in dataset.schema.names
    ],
})

pd.set_option("display.max_rows", None)
print(dict_df)
dict_df.to_csv("first_american_dictionary.csv", index=False)

# cook = dataset.to_table(
#     filter=ds.field("fips") == "17031"
# ).to_pandas()

cook = dataset.to_table(
    filter=ds.field("FIPS") == "17031"
).to_pandas()

print(len(cook))
print(cook.head())
print(cook["Year"].value_counts().sort_index())
print(cook["PropertyClassID"].value_counts())