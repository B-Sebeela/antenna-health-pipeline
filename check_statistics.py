
from src.storage.parquet import load_parquet
import numpy as np
import pandas as pd


scenarios = [
    "MF01_L_PLUS",
    "MF02_L_MINUS",
]


# Load healthy reference
nominal = load_parquet(
    "data/processed/H01_NOM/farfield.parquet"
)

nominal = nominal.set_index(["theta", "phi"])


print("RADIATION PATTERN DIFFERENCE DISTRIBUTION")
print("=" * 80)


for scenario in scenarios:

    df = load_parquet(
        f"data/processed/{scenario}/farfield.parquet"
    )

    df = df.set_index(["theta", "phi"])

    comparison = pd.DataFrame({
        "nominal": nominal["gain_total"],
        "fault": df["gain_total"],
    })

    comparison["difference"] = (
        comparison["fault"]
        - comparison["nominal"]
    )

    absolute_difference = comparison["difference"].abs()

    print()
    print(f"SCENARIO: {scenario}")
    print("-" * 80)

    print(
        f"Mean signed difference:       "
        f"{comparison['difference'].mean():.4f} dB"
    )

    print(
        f"Mean absolute difference:     "
        f"{absolute_difference.mean():.4f} dB"
    )

    print(
        f"Median absolute difference:   "
        f"{absolute_difference.median():.4f} dB"
    )

    print(
        f"Maximum absolute difference:  "
        f"{absolute_difference.max():.4f} dB"
    )

    print()

    thresholds = [
        0.1,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    print("Percentage of far-field points exceeding threshold:")

    for threshold in thresholds:

        percentage = (
            (absolute_difference >= threshold).mean()
            * 100
        )

        print(
            f"  >= {threshold:3.1f} dB : "
            f"{percentage:6.2f}%"
        )

