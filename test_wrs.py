# debug_wrs.py -- quick check for why calc_wrs returns 0
import pandas as pd
import itertools
import numpy as np
from scipy.stats import ttest_ind
from pathlib import Path

CSV = "data/input/synthetic_chain.csv"   # <- change this to the path you used
df = pd.read_csv(CSV)
print("File:", CSV)
print("Columns:", list(df.columns))
print("\nFirst 6 rows:\n", df.head(6).to_string(index=False))

# candidate protected columns to try (add any other names you use)
candidate_protected = ["Z1", "Z", "gender", "protected", "stock_symbol"]
found_protected = [c for c in candidate_protected if c in df.columns]
print("\nFound protected columns (from candidates):", found_protected)

# try to infer output columns of the form s1_m11 or s1_m11 / s1_m1 etc.
possible_output_cols = [c for c in df.columns if ("s1" in c or "s2" in c or c.startswith("s")) and ("m" in c)]
print("\nLikely output columns (sample):", possible_output_cols[:50])

# If you expect names like s1_m11 exactly, list those explicitly and check
expected_cols = [
    "s1_m11", "s1_m12", "s2_m21", "s2_m22",
    # also try variants with different separators your runner produced
    "call_model___d1__s1__m11", "call_model__d1__s1__m11"
]
present_expected = [c for c in expected_cols if c in df.columns]
print("\nExpected output columns present:", present_expected)

# helper to compute pairwise ttests and group sizes
def pairwise_tests(df, protected_col, output_col, min_per_group=2):
    grouped = df.groupby(protected_col)[output_col].apply(lambda x: x.dropna().values).to_dict()
    groups = list(grouped.keys())
    print(f"\nTesting output_col='{output_col}' with protected_col='{protected_col}' -> groups: {groups}")
    if len(groups) < 2:
        print("  <only one group found; cannot test>")
        return
    for g in groups:
        print(f"  group '{g}': n={len(grouped[g])}, sample (up to 5): {grouped[g][:5]}")
    for g1, g2 in itertools.combinations(groups, 2):
        d1, d2 = grouped[g1], grouped[g2]
        if len(d1) < min_per_group or len(d2) < min_per_group:
            print(f"  pair ({g1}, {g2}) skipped: group sizes {len(d1)} / {len(d2)} < min_per_group={min_per_group}")
            continue
        stat, p = ttest_ind(d1, d2, equal_var=False)
        print(f"  pair ({g1} vs {g2}): t={stat:.4f}, p={p:.4e}")

# Run tests on combinations we can find
protected_to_test = found_protected if found_protected else ([c for c in df.columns if c.lower().startswith("z")] or [])
if not protected_to_test:
    print("\nNo obvious protected column found. Add the correct protected_col name to candidate_protected or call me with the column name.")
else:
    # choose output columns to test: prefer expected, else fallback to possible_output_cols
    outputs = present_expected if present_expected else possible_output_cols[:20]
    if not outputs:
        print("\nNo output-like columns detected. Make sure your dataset has columns like 's1_m11', 's2_m21', etc.")
    else:
        for pcol in protected_to_test:
            for out in outputs:
                if out in df.columns:
                    pairwise_tests(df, pcol, out, min_per_group=2)
