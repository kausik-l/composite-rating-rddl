import pandas as pd
import numpy as np
from scipy import stats

def analyze_significance(file_path, scenario_name, num_stages):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {scenario_name} ({num_stages} Stages)")
    print(f"Source: {file_path}")
    print(f"{'='*80}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
        return

    # 1. REMOVE ZOMBIE ROWS
    df = df[df['Action'] != 'None']

    # 2. CALCULATE EPISODE TOTALS
    # We sum the step rewards to get the Total Episode Reward
    episode_stats = df.groupby(['Agent', 'Episode'])['Reward_Step'].sum().reset_index()

    # 3. NORMALIZE TO MATCH TABLE 1 (Per Stage Reward)
    # Table 1 values = Episode Total / Num Stages
    episode_stats['Reward_Per_Stage'] = episode_stats['Reward_Step'] / num_stages

    # Define our Champion
    champion = "Q-Learning (Combined)"
    
    # Check if Champion exists
    if champion not in episode_stats['Agent'].unique():
        print(f"Winner '{champion}' not found.")
        return

    # Get Champion Data
    champ_scores = episode_stats[episode_stats['Agent'] == champion]['Reward_Per_Stage']
    champ_mean = champ_scores.mean()
    champ_std = champ_scores.std()

    print(f"\n>>> CHAMPION: {champion}")
    print(f"    Mean Reward (Per Stage): {champ_mean:.4f} (Matches Table 1)")
    print(f"    Std Dev: {champ_std:.4f}")

    print(f"\n{'-'*30} STATISTICAL COMPARISONS {'-'*30}")
    
    # Compare against ALL other agents found in the CSV
    all_agents = episode_stats['Agent'].unique()
    others = [a for a in all_agents if a != champion]
    
    # Sort them so Q-Learning ones appear first
    others.sort(key=lambda x: "Q-Learning" not in x) 

    for opponent in others:
        opp_scores = episode_stats[episode_stats['Agent'] == opponent]['Reward_Per_Stage']
        opp_mean = opp_scores.mean()
        
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(champ_scores, opp_scores, equal_var=False)
        
        # Improvement %
        # Note: For negative numbers, standard % formula can be tricky.
        # We use simple diff here.
        diff = champ_mean - opp_mean
        
        # Significance Markers
        sig = "ns"
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"

        print(f"\nvs. {opponent:<25}")
        print(f"    Mean: {opp_mean:.4f} | Diff: {diff:+.4f}")
        print(f"    P-Value: {p_val:.2e}  [{sig}]")

# ==========================================================
# EXECUTION
# ==========================================================
# Make sure to update the filenames if they are different
files = [
    # (Name, Filename, Number_of_Stages)
    ("Small Scale", "results/10_05_plots/agent_trace_metrics.csv", 10),
    ("Large Scale", "results/30_15_plots/2_agent_trace_metrics.csv", 50) 
]

for name, fname, stages in files:
    analyze_significance(fname, name, stages)