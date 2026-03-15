import pandas as pd
import numpy as np
import os

from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def analyze_anova_tukey(file_path, scenario_name, num_stages=2):
    """
    Runs one-way ANOVA + Tukey HSD on Reward_Per_Stage across Agent groups.
    Mirrors the structure of your original script (champion selection + readable printing).
    """

    print(f"\n{'='*80}")
    print(f"ANOVA + TUKEY HSD: {scenario_name}")
    print(f"Source: {file_path}")
    print(f"NUM_STAGES = {num_stages}")
    print(f"{'='*80}")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return

    # 1) DATA CLEANING: remove zombie rows
    if "Action" not in df.columns or "Reward_Step" not in df.columns:
        print("[ERROR] Missing required columns. Need at least: Action, Reward_Step.")
        return
    df = df[df["Action"] != "None"]

    # 2) AGGREGATION: episode totals
    required = {"Agent", "Episode", "Reward_Step"}
    if not required.issubset(df.columns):
        print(f"[ERROR] Missing required columns: {sorted(required - set(df.columns))}")
        return

    episode_stats = (
        df.groupby(["Agent", "Episode"], as_index=False)["Reward_Step"]
          .sum()
          .rename(columns={"Reward_Step": "Episode_Reward"})
    )

    # 3) USE EPISODE TOTAL: reward per episode (no division by num_stages)
    episode_stats["Reward_Per_Stage"] = episode_stats["Episode_Reward"]

    # 4) IDENTIFY CHAMPION (same logic as your original code)
    champion = (
        episode_stats
        .groupby("Agent")["Reward_Per_Stage"]
        .mean()
        .idxmax()
    )

    champ_scores = episode_stats.loc[episode_stats["Agent"] == champion, "Reward_Per_Stage"]
    print(f"\n>>> CHAMPION AGENT: {champion}")
    print(f"    Mean Reward (Per Stage): {champ_scores.mean():.4f}")
    print(f"    Std Dev:                 {champ_scores.std():.4f}")
    print(f"    Sample Size (N):         {len(champ_scores)}")

    # 5) ONE-WAY ANOVA (omnibus)
    # Model: Reward_Per_Stage ~ C(Agent)
    model = ols("Reward_Per_Stage ~ C(Agent)", data=episode_stats).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(f"\n{'-'*30} ONE-WAY ANOVA (OMNIBUS) {'-'*30}")
    print(anova_table)

    if "C(Agent)" not in anova_table.index:
        print("[ERROR] ANOVA table missing C(Agent).")
        return

    p_anova = anova_table.loc["C(Agent)", "PR(>F)"]
    print(f"\nANOVA p-value for Agent effect: {p_anova:.3e}")

    # 6) TUKEY HSD (post-hoc, FWER-controlled)
    print(f"\n{'-'*30} TUKEY HSD (ALL PAIRS, FWER=0.05) {'-'*30}")
    tukey = pairwise_tukeyhsd(
        endog=episode_stats["Reward_Per_Stage"].values,
        groups=episode_stats["Agent"].values,
        alpha=0.05
    )
    print(tukey.summary())

    # 7) Champion-only view (like your original output)
    print(f"\n{'-'*30} CHAMPION VS OTHERS (TUKEY p-adj) {'-'*30}")

    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    # Filter for rows involving champion
    champ_rows = tukey_df[(tukey_df["group1"] == champion) | (tukey_df["group2"] == champion)].copy()

    if champ_rows.empty:
        print("[WARN] No champion comparisons found in Tukey table (unexpected).")
        return

    # Make a consistent "Opponent" column and "Champion - Opponent" direction
    def _opponent(row):
        return row["group2"] if row["group1"] == champion else row["group1"]

    champ_rows["Opponent"] = champ_rows.apply(_opponent, axis=1)

    # Tukey's meandiff is group2 - group1 (by statsmodels convention).
    # Convert to (Champion - Opponent) so it matches your old "diff = champ_mean - opp_mean".
    def _champ_minus_opp(row):
        g1, g2 = row["group1"], row["group2"]
        meandiff = float(row["meandiff"])
        # If row is (Champion, Opponent): meandiff = Opponent - Champion => Champion - Opponent = -meandiff
        # If row is (Opponent, Champion): meandiff = Champion - Opponent => Champion - Opponent = +meandiff
        return -meandiff if g1 == champion else meandiff

    champ_rows["Diff(Champion-Opp)"] = champ_rows.apply(_champ_minus_opp, axis=1)

    # Pretty print in your style
    champ_rows = champ_rows[["Opponent", "Diff(Champion-Opp)", "p-adj", "lower", "upper", "reject"]]
    champ_rows = champ_rows.sort_values("Opponent")

    print(f"{'OPPONENT':<30} | {'DIFF':<10} | {'P-ADJ':<10} | {'REJECT'}")
    print("-" * 70)
    for _, r in champ_rows.iterrows():
        opponent = str(r["Opponent"])
        diff = float(r["Diff(Champion-Opp)"])
        p_adj = float(r["p-adj"])
        reject = bool(r["reject"])
        sig = "*" if reject else "ns"
        print(f"{opponent:<30} | {diff:<+10.4f} | {p_adj:<10.3e} | {sig}")

    print(f"{'='*80}\n")


# ==========================================================
# EXECUTION BLOCK
# ==========================================================
if __name__ == "__main__":
    files = [
        ("WRS Only Mode", "../results/rts-small/agent_trace_metrics_wrs.csv"),
        ("Full Suite (WRS/DIE/BOTH)", "../results/rts-large/agent_trace_metrics_full.csv"),
    ]

    for name, fname in files:
        analyze_anova_tukey(fname, name, num_stages=2)
