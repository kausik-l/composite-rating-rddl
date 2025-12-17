import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from env.sentiment_small_env import SentimentPipelineEnv
from planner.sentiment_policy import ContextAwareQPlanner
from planner.sentiment_baselines import RandomPipelinePlanner, FixedPipelinePlanner, LookaheadFairnessPlanner
from env.metric_utils import calc_wrs

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # You can add "DIE" and "BOTH" here to train 3 different agents
    "ACTIVE_MODES": ["WRS"], 
    "TRAIN_EPISODES": 500,
    "EVAL_EPISODES": 500,
    "TRACE_LOG_EPISODES": 500, 
    "DATA_PATH": os.path.join(root, "composite-rating-rddl", "data", "input", "real_world", "master_sentiment_unibot.csv"),
    "DOMAIN_PATH": os.path.join(root, "composite-rating-rddl", "domains", "sentiment_pipeline.rddl"),
    "INSTANCE_PATH": os.path.join(root, "composite-rating-rddl", "instances", "sentiment_instance.rddl")
}

def run_experiment_trace():
    print(f"=== Sentiment Pipeline Trace (Train on specific -> Eval on BOTH) ===")
    
    if not os.path.exists(CONFIG["DATA_PATH"]):
        print(f"[ERROR] Data not found: {CONFIG['DATA_PATH']}")
        return

    # Initialize Env (Default start mode, will change in loop)
    env = SentimentPipelineEnv(CONFIG["DOMAIN_PATH"], CONFIG["INSTANCE_PATH"], CONFIG["DATA_PATH"], reward_mode="WRS")
    
    stage_map = {
        's1': ['trans_none', 'trans_danish', 'trans_spanish'],
        's2': ['m_bf', 'm_dbert', 'm_gru', 'm_random', 'm_textblob']
    }
    env.stage_model_map = stage_map
    
    # 1. Build Agents
    agents = []
    
    # Dynamically create Q-Learning Agents based on ACTIVE_MODES
    colors = {"WRS": "blue", "DIE": "green", "BOTH": "orange"}
    
    for mode in CONFIG["ACTIVE_MODES"]:
        agents.append({
            "name": f"Q-Learning ({mode})",
            "agent": ContextAwareQPlanner(env.action_space, stage_map, alpha=0.1, gamma=0.9, epsilon=0.2, action_name="select_component"),
            "learns": True,
            "train_mode": mode, # Mode used for training
            "color": colors.get(mode, "black")
        })

    # Baselines (Evaluated under BOTH mode logic)
    # Note: 'train_mode' is irrelevant for them as they don't learn, but we set it for consistency
    agents.extend([
        {"name": "Heuristic", "agent": LookaheadFairnessPlanner(stage_map, env, action_name="select_component"), "learns": False, "train_mode": "BOTH", "color": "red"},
        {"name": "Fixed (Biased)", "agent": FixedPipelinePlanner(stage_map, 0, action_name="select_component"), "learns": False, "train_mode": "BOTH", "color": "purple"},
        {"name": "Random", "agent": RandomPipelinePlanner(stage_map, action_name="select_component"), "learns": False, "train_mode": "BOTH", "color": "gray"}
    ])
    
    # Trace File Setup
    csv_trace_path = "agent_trace_metrics_combined.csv"
    if os.path.exists(csv_trace_path):
        os.remove(csv_trace_path)
        print(f"  > Cleared previous trace file: {csv_trace_path}")

    results_table = []
    plot_data = {}

    # 2. Execution Loop
    for entry in agents:
        name = entry['name']
        agent = entry['agent']
        learns = entry['learns']
        train_mode = entry['train_mode']
        color = entry.get('color', 'black')
        
        print(f"\n--- Processing Agent: {name} ---")
        
        # ---------------------------------------------------------
        # A. TRAINING PHASE (Uses specific 'train_mode')
        # ---------------------------------------------------------
        if learns:
            print(f"  > Training with Reward Mode: {train_mode}")
            env.reward_mode = train_mode # Set env to WRS, DIE, or BOTH specifically for learning
            
            for _ in tqdm(range(CONFIG["TRAIN_EPISODES"]), desc="Train"):
                state, _ = env.reset()
                while True:
                    action = agent.sample_action(state)
                    next_state, r, done, _, _ = env.step(action)
                    agent.update(state, action, r, next_state)
                    state = next_state
                    if done: break
                agent.epsilon = max(0.05, agent.epsilon * 0.995)
        
        # ---------------------------------------------------------
        # B. EVALUATION PHASE (ALWAYS Uses 'BOTH')
        # ---------------------------------------------------------
        print(f"  > Switching Env Mode to 'BOTH' for Evaluation/Tracing...")
        env.reward_mode = "BOTH" 
        
        current_agent_trace = []
        
        # Lists for calculating averages
        rewards = []
        costs = []
        raw_wrs_list = []
        raw_die_list = []
        planning_times = []
        pipeline_choices = []
        
        eval_pbar = tqdm(range(CONFIG["EVAL_EPISODES"]), desc=f"Eval {name}")
        
        for ep_idx in eval_pbar:
            state, _ = env.reset()
            stage_count = 1
            total_rew, ep_cost, ep_raw_wrs, ep_raw_die = 0, 0, 0, 0
            current_pipeline = []
            
            while True:
                t0 = time.time()
                action = agent.sample_action(state)
                t1 = time.time()
                planning_times.append(t1 - t0)
                
                selected_model_name = "None"
                for k, v in action.items():
                    if v == 1 and "select_component" in k:
                        selected_model_name = k.split("___")[-1]
                        current_pipeline.append(selected_model_name)
                        break
                
                next_state, r, done, _, info = env.step(action)
                
                total_rew += r
                
                metrics = info.get('metrics', {})
                if 'metrics' in info:
                    ep_cost += info['metrics'].get('rddl_cost', 0)
                    ep_raw_wrs += info['metrics'].get('raw_wrs', 0.0)
                    ep_raw_die += info['metrics'].get('raw_die', 0.0)
                
                if ep_idx < CONFIG["TRACE_LOG_EPISODES"]:
                    current_agent_trace.append({
                        "Agent": name,
                        "Episode": ep_idx,
                        "Stage": stage_count,
                        "Action": selected_model_name,
                        "Reward_Step": r, # This reward now reflects 'BOTH' penalties
                        "Raw_WRS": metrics.get('raw_wrs', 0.0),
                        "Raw_DIE": metrics.get('raw_die', 0.0),
                        "Penalty_Fairness": metrics.get('wrs_penalty', 0.0)
                    })
                
                state = next_state
                stage_count += 1
                if done: break
            
            rewards.append(total_rew)
            costs.append(ep_cost)
            raw_wrs_list.append(ep_raw_wrs)
            raw_die_list.append(ep_raw_die)
            pipeline_choices.append(tuple(current_pipeline))

        # Save Trace
        write_header = not os.path.exists(csv_trace_path) or os.path.getsize(csv_trace_path) == 0
        df_chunk = pd.DataFrame(current_agent_trace)
        if not df_chunk.empty:
            df_chunk.to_csv(csv_trace_path, mode='a', header=write_header, index=False)
            
        # Store Plot Data
        plot_data[name] = (rewards, color)
        
        # Calculate Top 3 Pipelines
        recent_choices = pipeline_choices[-500:]
        top3_str = "N/A"
        if recent_choices:
            counts = Counter(recent_choices).most_common(3)
            parts = []
            for i, (pipe, count) in enumerate(counts):
                pipe_str = " -> ".join(pipe)
                freq_pct = (count / len(recent_choices)) * 100
                parts.append(f"[{i+1}: {pipe_str} ({freq_pct:.1f}%)]")
            top3_str = " ".join(parts)

        # Store Results Row
        row = {
            "Agent": name,
            "Train Mode": train_mode if learns else "N/A",
            "Eval Mode": "BOTH",
            "Avg Reward": np.mean(rewards),
            "Avg Cost": np.mean(costs),
            "Avg WRS": np.mean(raw_wrs_list),
            "Avg DIE": np.mean(raw_die_list),
            "Time (ms)": np.mean(planning_times) * 1000,
            "Top 3 Pipelines": top3_str
        }
        results_table.append(row)

    print("\n=== Trace Generation Complete ===")
    
    # 3. Output Table
    df = pd.DataFrame(results_table)
    print("\n=== FINAL SENTIMENT RESULTS ===")
    cols = ["Agent", "Train Mode", "Eval Mode", "Avg Reward", "Avg WRS", "Avg DIE", "Top 3 Pipelines"]
    df_disp = df[cols].copy()
    pd.set_option('display.max_colwidth', None)
    print(df_disp.to_string(index=False))
    df.to_csv("sentiment_results_table_combined.csv", index=False)
    
    # 4. Generate Plot
    plt.figure(figsize=(12, 6))
    for name, (rew, col) in plot_data.items():
        window = 20
        if len(rew) > window:
            smoothed = np.convolve(rew, np.ones(window)/window, mode='valid')
            lw = 2.5 if "Q-Learning" in name else 1.5
            plt.plot(smoothed, label=name, color=col, linewidth=lw)
        else:
             plt.plot(rew, label=name, color=col)
    
    plt.title("Evaluation under 'BOTH' mode (WRS + DIE)")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Total Reward (Penalized by WRS + DIE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rts_small_comparison_plot.png")
    print("\nSaved plot to 'rts_small_comparison_plot.png'")

if __name__ == "__main__":
    run_experiment_trace()