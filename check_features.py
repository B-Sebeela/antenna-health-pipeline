
import pandas as pd
import numpy as np


FEATURE_PATH = "data/features/scenario_features.parquet"


df = pd.read_parquet(FEATURE_PATH)


print("=" * 80)
print("FEATURE DATASET INSPECTION")
print("=" * 80)

print()
print("Dataset shape:")
print(f"  Rows:    {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")

print()
print("Scenario IDs:")
for scenario in df["scenario_id"]:
    print(f"  - {scenario}")


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing = df.isna().sum()

missing_features = missing[missing > 0]

if missing_features.empty:
    print("No missing values found.")
else:
    print(missing_features)


# ---------------------------------------------------------------------------
# Constant features
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("CONSTANT FEATURES")
print("=" * 80)

numeric_df = df.select_dtypes(include=np.number)

constant_features = [
    column
    for column in numeric_df.columns
    if numeric_df[column].nunique() <= 1
]

if constant_features:
    print(f"{len(constant_features)} constant features found:")

    for feature in constant_features:
        print(f"  - {feature}")
else:
    print("No constant features found.")


# ---------------------------------------------------------------------------
# Feature ranges
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("FEATURE RANGES")
print("=" * 80)

feature_ranges = pd.DataFrame({
    "min": numeric_df.min(),
    "max": numeric_df.max(),
    "mean": numeric_df.mean(),
    "std": numeric_df.std(),
})

print(feature_ranges.to_string())


# ---------------------------------------------------------------------------
# Highly correlated features
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("HIGHLY CORRELATED FEATURES")
print("=" * 80)

correlation = numeric_df.corr()

pairs = []

columns = correlation.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        value = correlation.iloc[i, j]

        if abs(value) >= 0.95:
            pairs.append(
                (
                    columns[i],
                    columns[j],
                    value,
                )
            )

if pairs:

    pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    for feature_a, feature_b, value in pairs:
        print(
            f"{feature_a:45} "
            f"{feature_b:45} "
            f"corr={value:.4f}"
        )

else:
    print("No feature pairs with |correlation| >= 0.95 found.")


print()
print("=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

