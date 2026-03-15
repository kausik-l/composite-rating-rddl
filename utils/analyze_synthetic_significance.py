import pandas as pd
import numpy as np

from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def analyze_anova_tukey(file_path, scenario_name, num_stages, champion="Q-Learning (Combined)"):
    print(f"\n{'='*80}")
    print(f"ANOVA + TUKEY HSD: {scenario_name} ({num_stages} Stages)")
    print(f"Source: {file_path}")
    print(f"{'='*80}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
        return

    # 1) REMOVE ZOMBIE ROWS
    df = df[df["Action"] != "None"]

    # 2) EPISODE TOTALS
    episode_stats = df.groupby(["Agent", "Episode"], as_index=False)["Reward_Step"].sum()

    # 3) NORMALIZE TO PER-STAGE REWARD
    episode_stats["Reward_Per_Stage"] = episode_stats["Reward_Step"] / num_stages

    agents = episode_stats["Agent"].unique()
    if len(agents) < 2:
        print("Not enough agent groups for ANOVA.")
        return

    if champion not in agents:
        print(f"Champion '{champion}' not found. Proceeding without champion highlighting.")
        champion = None

    # --------------------------
    # A) One-way ANOVA
    # --------------------------
    # Model: Reward_Per_Stage ~ C(Agent)
    model = ols("Reward_Per_Stage ~ C(Agent)", data=episode_stats).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n--- ONE-WAY ANOVA (omnibus test) ---")
    print(anova_table)

    # Extract p-value for Agent effect (whether any group differs)
    p_anova = anova_table.loc["C(Agent)", "PR(>F)"]
    print(f"\nANOVA p-value for Agent effect: {p_anova:.3e}")

    # --------------------------
    # B) Tukey HSD post-hoc
    # --------------------------
    print("\n--- TUKEY HSD (pairwise, FWER-controlled) ---")
    tukey = pairwise_tukeyhsd(
        endog=episode_stats["Reward_Per_Stage"].values,
        groups=episode_stats["Agent"].values,
        alpha=0.05
    )

    # Full table
    print(tukey.summary())

    # Optional: highlight champion comparisons only
    if champion is not None:
        print(f"\n--- TUKEY RESULTS: comparisons involving champion '{champion}' ---")
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        )

        champ_rows = tukey_df[(tukey_df["group1"] == champion) | (tukey_df["group2"] == champion)]
        if champ_rows.empty:
            print("No champion comparisons found in Tukey output (unexpected).")
        else:
            # Sort for readability
            champ_rows = champ_rows.sort_values(["group1", "group2"])
            print(champ_rows.to_string(index=False))


# ==========================================================
# EXECUTION
# ==========================================================
files = [
    ("Small Scale", "results/10_05_plots/agent_trace_metrics.csv", 10),
    ("Large Scale", "results/30_15_plots/2_agent_trace_metrics.csv", 50),
]

for name, fname, stages in files:
    analyze_anova_tukey(fname, name, stages, champion="Q-Learning (Combined)")
