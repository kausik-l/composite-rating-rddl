import itertools
import numpy as np
from scipy.stats import ttest_ind

def calc_wrs(df, protected_col="gender", output_col="sentiment_outcome",
             alphas=(0.05, 0.30, 0.40), weights=(1.0, 0.7, 0.6),
             alternative="two-sided", min_per_group=2):
    grouped = df.groupby(protected_col)[output_col].apply(lambda x: x.dropna().values).to_dict()
    groups = list(grouped.keys())
    k = len(groups)

    if k < 2:
        return 0.0

    wrs = 0.0
    pvals = {}

    for g1, g2 in itertools.combinations(groups, 2):
        d1, d2 = grouped[g1], grouped[g2]
        if len(d1) < min_per_group or len(d2) < min_per_group:
            continue
        stat, p = ttest_ind(d1, d2, equal_var=False, alternative=alternative)
        pvals[(g1, g2)] = p

    for alpha, w in zip(alphas, weights):
        for p in pvals.values():
            if np.isfinite(p) and p < alpha:
                wrs += w

    return round(wrs, 2)
