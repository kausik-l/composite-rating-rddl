# diagnostic_wrs.py
import pandas as pd
import numpy as np
import itertools
from scipy.stats import t

def diagnostic_wrs_student(df, protected_col, output_col, min_per_group=2):
    print(f"Diagnostic for protected='{protected_col}', output='{output_col}'\n")
    if protected_col not in df.columns:
        print("-> Protected column not found.")
        return
    if output_col not in df.columns:
        print("-> Output column not found.")
        return

    sub = df[[protected_col, output_col]].dropna()
    if sub.shape[0] == 0:
        print("-> No rows after dropping NaNs.")
        return

    z = sub[protected_col]
    y = sub[output_col].astype(float)

    # show unique counts
    levels = list(pd.unique(z))
    print("All observed levels (count):")
    for lvl in levels:
        cnt = int((z == lvl).sum())
        mean = None
        var = None
        if cnt>0:
            mean = float(y[z==lvl].mean())
            var = float(y[z==lvl].var(ddof=1)) if cnt>1 else float('nan')
        print(f"  Level={repr(lvl):20} count={cnt:4} mean={mean} var(ddof=1)={var}")
    print()

    # Filter to levels with enough data
    valid_levels = [lvl for lvl in levels if int((z==lvl).sum()) >= min_per_group]
    print("Levels with >= min_per_group:", valid_levels)
    if len(valid_levels) < 2:
        print("-> Fewer than 2 valid levels; no pairwise tests will be run.")
        return

    # examine pairwise
    pairs = list(itertools.combinations(valid_levels, 2))
    print("\nPairwise checks:")
    for a,b in pairs:
        y_a = y[z==a].dropna().astype(float)
        y_b = y[z==b].dropna().astype(float)
        n_a = len(y_a); n_b = len(y_b)
        var_a = y_a.var(ddof=1) if n_a>1 else float('nan')
        var_b = y_b.var(ddof=1) if n_b>1 else float('nan')
        pooled_df = n_a + n_b - 2
        pooled_var = None
        degenerate = False
        if pooled_df > 0 and not np.isnan(var_a) and not np.isnan(var_b):
            pooled_var = ((n_a - 1)*var_a + (n_b - 1)*var_b) / pooled_df
            if pooled_var <= 0 or np.isnan(pooled_var):
                degenerate = True
        else:
            degenerate = True
        print(f" Pair {repr(a)} vs {repr(b)}: n={n_a},{n_b} var={var_a},{var_b} pooled_var={pooled_var} degenerate={degenerate}")

    print("\nIf many pairs are degenerate or levels have tiny counts, that's why WRS==0.0.")



df = pd.read_csv("data/input/real_world/unibot/eng/bf/bf.csv") 
diagnostic_wrs_student(df, 'User_gender', 'Sentiment', min_per_group=2)