import pandas as pd
import numpy as np
from scipy import stats
import os

def analyze_significance(file_path, scenario_name):
    # 1. Pipeline Length Definition
    # For Sentiment Task: Stage 1 (Translate) -> Stage 2 (Model)
    NUM_STAGES = 2 

    print(f"\n{'='*80}")
    print(f"ANALYZING: {scenario_name}")
    print(f"Source: {file_path}")
    print(f"{'='*80}")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return

    # 2. DATA CLEANING
    # Remove "Zombie" stages (Action == 'None') caused by RDDL horizon padding
    df = df[df['Action'] != 'None']

    # 3. AGGREGATION (Episode Totals)
    # We sum the rewards for each episode to get Independent Samples (N=500)
    episode_stats = df.groupby(['Agent', 'Episode'])['Reward_Step'].sum().reset_index()

    # 4. NORMALIZE (Reward Per Stage)
    episode_stats['Reward_Per_Stage'] = episode_stats['Reward_Step'] / NUM_STAGES

    # 5. IDENTIFY CHAMPION (Auto-Detect)
    # Priority: Combined -> WRS -> DIE
    available_agents = episode_stats['Agent'].unique()
    
    if "Q-Learning (Combined)" in available_agents:
        champion = "Q-Learning (Combined)"
    elif "Q-Learning (WRS)" in available_agents:
        champion = "Q-Learning (WRS)"
    elif "Q-Learning (DIE)" in available_agents:
        champion = "Q-Learning (DIE)"
    else:
        # Fallback if names changed
        q_agents = [a for a in available_agents if "Q-Learning" in a]
        champion = q_agents[0] if q_agents else available_agents[0]

    # Get Champion Data
    champ_scores = episode_stats[episode_stats['Agent'] == champion]['Reward_Per_Stage']
    champ_mean = champ_scores.mean()
    champ_std = champ_scores.std()

    print(f"\n>>> CHAMPION AGENT: {champion}")
    print(f"    Mean Reward (Per Stage): {champ_mean:.4f}")
    print(f"    Std Dev:                 {champ_std:.4f}")
    print(f"    Sample Size (N):         {len(champ_scores)}")

    print(f"\n{'-'*30} STATISTICAL COMPARISONS {'-'*30}")
    print(f"{'OPPONENT':<30} | {'MEAN':<10} | {'DIFF':<10} | {'P-VALUE':<10} | {'SIG'}")
    print("-" * 80)

    # Sort opponents: Other Q-Learners first, then Baselines
    others = [a for a in available_agents if a != champion]
    others.sort(key=lambda x: "Q-Learning" not in x) 

    for opponent in others:
        opp_scores = episode_stats[episode_stats['Agent'] == opponent]['Reward_Per_Stage']
        opp_mean = opp_scores.mean()
        
        # --- WELCH'S T-TEST ---
        # equal_var=False handles the fact that Heuristic/Random often have higher variance
        t_stat, p_val = stats.ttest_ind(champ_scores, opp_scores, equal_var=False)
        
        # Calculate Difference
        diff = champ_mean - opp_mean
        
        # Improvement Percentage (Handling negative rewards)
        # If Reward is -10 vs -15, Diff is +5. Improvement is 5 / |-15| = 33%
        pct_imp = (diff / abs(opp_mean)) * 100

        # Significance Markers
        sig = "ns"
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"

        # Color coding for terminal output (Optional)
        # Green if Champion Won, Red if Champion Lost significantly
        result_str = f"{opponent:<30} | {opp_mean:<10.4f} | {diff:<+10.4f} | {p_val:.2e}   | {sig}"
        print(result_str)
        
        if p_val < 0.05:
            if diff > 0:
                print(f"    -> RESULT: Champion is BETTER by {pct_imp:.1f}% (Significant)")
            else:
                print(f"    -> RESULT: Champion is WORSE by {abs(pct_imp):.1f}% (Significant)")

    print(f"{'='*80}\n")

# ==========================================================
# EXECUTION BLOCK
# ==========================================================
if __name__ == "__main__":
    # Define your files here
    files = [
        ("WRS Only Mode", "results/rts-small/agent_trace_metrics_wrs.csv"),
        ("Full Suite (WRS/DIE/BOTH)", "results/rts-large/agent_trace_metrics_full.csv") 
    ]

    for name, fname in files:
        analyze_significance(fname, name)