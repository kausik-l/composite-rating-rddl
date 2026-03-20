import pandas as pd
import os

def analyze_top_plans(file_path, top_n=3):
    """
    Reads an agent trace CSV, filters out 'None' actions, 
    and reports the Top-3 most frequent pipeline plans per agent.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    print(f"Loading trace file: {file_path}...")
    df = pd.read_csv(file_path)

    # 1. CLEANING
    # Ensure 'Action' is treated as string to avoid "float found" errors
    # This converts NaNs to "nan" string, which we can then filter out
    df['Action'] = df['Action'].astype(str)
    
    # Remove rows where Action is "None" or "nan" (case-insensitive coverage)
    df_clean = df[~df['Action'].str.lower().isin(['none', 'nan'])].copy()
    
    # Sort to ensure actions are in the correct stage order
    if 'Stage' in df_clean.columns:
        df_clean = df_clean.sort_values(by=['Agent', 'Episode', 'Stage'])

    # Get list of unique agents
    agents = df_clean['Agent'].unique()

    for agent in agents:
        print(f"\n{'='*60}")
        print(f"AGENT: {agent}")
        print(f"{'='*60}")
        
        # Filter for this specific agent
        agent_df = df_clean[df_clean['Agent'] == agent]
        
        # 2. GROUPING
        # Group by Episode to create the full plan tuple (e.g., ('m4', 'm2', 'm1'))
        plans = agent_df.groupby('Episode')['Action'].apply(tuple)
        
        total_episodes = len(plans)
        print(f"Total Episodes: {total_episodes}")
        
        # 3. RANKING
        # Get the top N most frequent plans
        top_plans = plans.value_counts().head(top_n)
        
        if top_plans.empty:
            print("  No valid plans found.")
            continue
            
        print(f"\n--- Top {top_n} Plans ---")
        for rank, (pipeline_tuple, count) in enumerate(top_plans.items(), 1):
            # Convert tuple to readable string
            pipeline_str = " -> ".join(pipeline_tuple)
            
            # Calculate Frequency %
            freq_percent = (count / total_episodes) * 100
            
            print(f"#{rank}: {pipeline_str}")
            print(f"    Frequency: {count} ({freq_percent:.1f}%)")
            print("-" * 30)

if __name__ == "__main__":
    # Change this filename to match your specific trace file
    # e.g., "agent_trace_metrics_30_15.csv" or "2_agent_trace_metrics.csv"
    csv_file = "results/30_15_plots/2_agent_trace_metrics.csv" 
    # csv_file = "results/30_15_plots/2_agent_trace_metrics.csv" 

    analyze_top_plans(csv_file, top_n=3)
    
