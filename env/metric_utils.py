import itertools
import numpy as np
from scipy.stats import ttest_ind

def calc_wrs(
    df,
    protected_col="gender",
    output_col="sentiment_outcome",
    alphas=(0.05, 0.30, 0.40),  
    weights=(1.0, 0.7, 0.6),     
    alternative="two-sided",   
    # need at least 2 samples in a group.  
    min_per_group=2              
):
    """
    Args:
        df: pandas DataFrame containing the data.
        protected_col: column with protected attribute (e.g., "gender").
        output_col: column with model outputs (e.g., "sentiment_outcome").
        alphas: list of significance levels for hypothesis tests.
        weights: corresponding weights for each alpha.
        alternative: 'two-sided' test (default).
        min_per_group: minimum rows per group required to run a test.

    Returns:
        wrs (float): Weighted Rejection Score. Higher = more bias detected.
    """

    # -------------------------------------------------------------
    # Group the data by protected attribute (e.g., male/female/neutral).
    # -------------------------------------------------------------
    grouped = (
        df.groupby(protected_col)[output_col]
        .apply(lambda x: x.dropna().values)
        .to_dict()
    )
    groups = list(grouped.keys())
    k = len(groups)

    # If we have fewer than 2 groups, WRS is meaningless -> return 0.
    if k < 2:
        return 0.0

    wrs = 0.0        
    pvals = {}       

    # -------------------------------------------------------------
    #    For every pair of groups, run t-test.
    # -------------------------------------------------------------
    for g1, g2 in itertools.combinations(groups, 2):
        d1, d2 = grouped[g1], grouped[g2]

        # Skip pairs if one group is too small.
        if len(d1) < min_per_group or len(d2) < min_per_group:
            continue

        # Welch's t-test: compare means of the two groups
        stat, p = ttest_ind(d1, d2, equal_var=False, alternative=alternative)
        pvals[(g1, g2)] = p

    # -------------------------------------------------------------
    # For each significance threshold (alpha),
    #    check if the difference between groups is "significant".
    #    If yes, then add the corresponding weight to WRS.
    # -------------------------------------------------------------
    for alpha, w in zip(alphas, weights):
        for p in pvals.values():
            if np.isfinite(p) and p < alpha:
                wrs += w

    return round(wrs, 2)
