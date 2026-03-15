# metric_utils_student_fixed.py
import math
import itertools
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import t

def _pooled_stats(d1: pd.Series, d2: pd.Series) -> Tuple[float, float, float, float, float]:
    """
    Return (m1, m2, pooled_var, n1, n2)
    pooled_var is computed with ddof=1 sample variances. If not computable, pooled_var <= 0.
    """
    x1 = d1.dropna().astype(float)
    x2 = d2.dropna().astype(float)
    n1 = len(x1); n2 = len(x2)
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan'), -1.0, n1, n2
    m1 = float(x1.mean()) if n1>0 else float('nan')
    m2 = float(x2.mean()) if n2>0 else float('nan')
    s1 = float(x1.var(ddof=1)) if n1 > 1 else 0.0
    s2 = float(x2.var(ddof=1)) if n2 > 1 else 0.0
    df = n1 + n2 - 2
    pooled_var = -1.0
    if df > 0:
        pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / df
    return m1, m2, pooled_var, n1, n2

def _pooled_t_stat_and_df(d1: pd.Series, d2: pd.Series) -> Tuple[float, float]:
    """
    Compute Student's pooled t-stat and df. If degenerate (pooled_var <= 0) returns (nan, df).
    """
    m1, m2, pooled_var, n1, n2 = _pooled_stats(d1, d2)
    df = float(max(1, n1 + n2 - 2))
    if pooled_var <= 0 or math.isnan(pooled_var):
        return float('nan'), df
    se = math.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
    if se == 0 or math.isnan(se):
        return float('nan'), df
    t_stat = (m1 - m2) / se
    return float(t_stat), df

def calc_wrs(
    df: pd.DataFrame,
    protected_cols,
    output_col: str,
    min_per_group: int = 2
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Student's pooled WRS with degenerate-case handling:
      - For each categorical protected attribute, do pairwise comparisons (k choose 2).
      - If pooled_var == 0 and means differ, treat as rejection at all alphas (add full weights sum).
      - Otherwise perform Student's t and compare to critical t for each two-sided alpha.
    Returns (wrs_total, details)
    """
    if isinstance(protected_cols, str):
        protected_cols = [protected_cols]

    alphas = [0.05, 0.30, 0.40]  # two-sided alphas (95%,70%,60% CIs)
    weights = [1.0, 0.7, 0.6]    # as in paper text

    if output_col not in df.columns:
        return 0.0, []

    wrs_total = 0.0
    details: List[Dict[str, Any]] = []
    full_pair_weight = sum(weights)

    for z_col in protected_cols:
        if z_col not in df.columns:
            continue

        sub = df[[z_col, output_col]].dropna()
        if sub.shape[0] == 0:
            continue

        z_raw = sub[z_col]
        y = sub[output_col].astype(float)

        levels = list(pd.unique(z_raw.dropna()))
        if len(levels) < 2:
            continue

        pairs = list(itertools.combinations(levels, 2))
        for (lvl_i, lvl_j) in pairs:
            yi = y[z_raw == lvl_i].dropna().astype(float)
            yj = y[z_raw == lvl_j].dropna().astype(float)
            ni = len(yi); nj = len(yj)
            if ni < min_per_group or nj < min_per_group:
                # skip very small groups (matches previous guard)
                continue

            m1, m2, pooled_var, _, _ = _pooled_stats(yi, yj)
            df_dof = float(max(1, ni + nj - 2))

            # Degenerate case: pooled_var <= 0
            if pooled_var <= 0 or math.isnan(pooled_var):
                # If means are equal (or both nan), treat non-significant
                if not (np.isfinite(m1) and np.isfinite(m2)):
                    continue
                if abs(m1 - m2) == 0.0:
                    # identical constants --> no difference
                    continue
                # If group means differ while pooled variance is zero, **treat as maximal evidence**
                # Add full weight sum for this pair and record as such.
                wrs_total += full_pair_weight
                details.append({
                    "protected_col": z_col,
                    "pair": (lvl_i, lvl_j),
                    "degenerate_pooled_var": float(pooled_var),
                    "means": (float(m1), float(m2)),
                    "action": "degenerate_means_differ -> add_full_weights",
                    "added_weight": float(full_pair_weight)
                })
                continue

            # Otherwise, compute t-stat and evaluate per-alpha
            t_stat, df_calc = _pooled_t_stat_and_df(yi, yj)
            if not np.isfinite(t_stat):
                continue
            for alpha, w in zip(alphas, weights):
                t_crit = t.ppf(1 - alpha / 2.0, df_calc)
                if abs(t_stat) > t_crit:
                    wrs_total += w
                    details.append({
                        "protected_col": z_col,
                        "pair": (lvl_i, lvl_j),
                        "t_stat": float(t_stat),
                        "dof": float(df_calc),
                        "alpha_two_sided": float(alpha),
                        "weight": float(w)
                    })

    return float(wrs_total)