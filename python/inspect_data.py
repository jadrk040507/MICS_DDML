"""
Script to inspect the MICS dataset and find household ID variable.
"""

import pandas as pd
from config import DATA_FILE

print("=" * 60)
print("INSPECTING MICS DATASET")
print("=" * 60)

# Load data
print(f"\nLoading data from: {DATA_FILE}")
dt = pd.read_stata(DATA_FILE, convert_categoricals=False)

print(f"\nTotal observations: {dt.shape[0]:,}")
print(f"Total variables: {dt.shape[1]}")

# Search for household ID variables
print("\n" + "=" * 60)
print("SEARCHING FOR HOUSEHOLD ID VARIABLES")
print("=" * 60)

# Common household ID patterns in MICS/Survey data
patterns = ['hh', 'household', 'cluster', 'psu', 'serial', 'id', 'sample']

found_vars = []
for pattern in patterns:
    matches = [col for col in dt.columns if pattern.lower() in col.lower()]
    found_vars.extend(matches)

# Remove duplicates
found_vars = list(set(found_vars))

if found_vars:
    print(f"\n✓ Found {len(found_vars)} potential household ID variables:\n")
    for var in sorted(found_vars):
        # Show sample values and unique count
        unique_vals = dt[var].nunique()
        sample_vals = dt[var].dropna().head(3).tolist()
        dtype = dt[var].dtype
        print(f"  {var:30s} | Type: {str(dtype):15s} | Unique: {unique_vals:>8,} | Sample: {sample_vals}")
else:
    print("\n✗ No household ID variables found with common patterns")

# Show all variables that might be IDs
print("\n" + "=" * 60)
print("ALL VARIABLES CONTAINING 'ID', 'HH', OR 'CLUSTER'")
print("=" * 60)

all_cols = dt.columns.tolist()
id_cols = [col for col in all_cols if any(x in col.lower() for x in ['id', 'hh', 'cluster', 'psu'])]

if id_cols:
    for var in sorted(id_cols):
        unique_vals = dt[var].nunique()
        dtype = dt[var].dtype
        print(f"  {var:30s} | Type: {str(dtype):15s} | Unique: {unique_vals:>8,}")
else:
    print("\nNo variables found with 'id', 'hh', or 'cluster' in name")

# Show first 30 columns
print("\n" + "=" * 60)
print("FIRST 30 VARIABLES IN DATASET")
print("=" * 60)
for i, col in enumerate(dt.columns[:30]):
    unique_vals = dt[col].nunique()
    print(f"  {i+1:2d}. {col:30s} | Unique: {unique_vals:>8,}")

# Show country variable info
print("\n" + "=" * 60)
print("COUNTRY INFORMATION")
print("=" * 60)
country_cols = [col for col in ['Country', 'country', 'country_cat'] if col in dt.columns]
if country_cols:
    for col in country_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {dt[col].nunique()}")
        print(f"  Sample values: {dt[col].dropna().unique()[:5].tolist()}")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("""
Based on typical MICS datasets, the household ID is usually one of:
  - hhid
  - household_id  
  - cluster
  - psu
  - sample_id

Look at the output above and choose the variable with:
  1. Many unique values (thousands, not dozens)
  2. Numeric or string type
  3. Name suggesting household/cluster ID

Then edit config.py:
  CLUSTER_VAR = "YOUR_VARIABLE_NAME"
""")
