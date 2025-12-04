import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure root is in path so we can import planner/env modules
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from real_world.sentiment_env import SentimentPipelineEnv
from planner.family_policy import ContextAwareQPlanner
from planner.baselines import RandomPipelinePlanner, FixedPipelinePlanner, LookaheadFairnessPlanner

def run_experiment():
    print("=== Sentiment Pipeline: Comprehensive Evaluation ===")
    
    # 1. Paths
    data_path = os.path.join(root, "data", "input", "real_world", "master_sentiment.csv") 
    domain_path = os.path.join(root, "domains", "sentiment_pipeline.rddl")
    inst_path = os.path.join(root, "instances", "sentiment_instance.rddl")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Master CSV not found at {data_path}")
        print("Please run 'utils/preprocess_real_world.py' first!")
        return

    # 2. Init Env
    env = SentimentPipelineEnv(domain_path, inst_path, data_path)
    
    # Define Stage Map manually for this custom domain
    # This tells the baselines/Q-learner which actions are valid at which stage
    stage_map = {
        's1': ['trans_none', 'trans_danish', 'trans_spanish'],
        's2': ['m_bf', 'm_dbert', 'm_gru', 'm_random', 'm_textblob']
    }
    env.stage_model_map = stage_map

    # 3. Define Agents
    # Note: Fixed(0) corresponds to trans_none / m_bf
    #       Fixed(1) corresponds to trans_danish / m_dbert
    # action_name="select_component" is crucial for RDDL compatibility
    agents = [
        {"name": "Context-Aware Q-Learning", "agent": ContextAwareQPlanner(env.action_space, stage_map, alpha=0.1, gamma=0.9, epsilon=0.2, action_name="select_component"), "learns": True, "color": "purple"},
        {"name": "Heuristic (Lookahead)", "agent": LookaheadFairnessPlanner(stage_map, env, action_name="select_component"), "learns": False, "color": "green"},
        {"name": "Fixed (Original/BF)", "agent": FixedPipelinePlanner(stage_map, 0, action_name="select_component"), "learns": False, "color": "red"},
        {"name": "Fixed (Danish/DBert)", "agent": FixedPipelinePlanner(stage_map, 1, action_name="select_component"), "learns": False, "color": "orange"},
        {"name": "Random", "agent": RandomPipelinePlanner(stage_map, action_name="select_component"), "learns": False, "color": "gray"}
    ]
    
    results_table = []
    plot_data = {}

    # 4. Run Loop
    episodes = 500
    
    for entry in agents:
        name = entry['name']
        agent = entry['agent']
        learns = entry['learns']
        color = entry['color']
        
        print(f"Running {name}...")
        rewards = []
        costs = []
        wrs_scores = []
        planning_times = []
        
        # Start Timer for Training Duration
        start_clock = time.time()
        
        for ep in tqdm(range(episodes)):
            state, _ = env.reset()
            total_rew = 0
            ep_cost = 0
            ep_wrs = 0
            
            while True:
                # Measure Planning Time (Inference)
                t0 = time.time()
                action = agent.sample_action(state)
                t1 = time.time()
                planning_times.append(t1 - t0)
                
                next_state, r, done, _, info = env.step(action)
                
                if learns:
                    agent.update(state, action, r, next_state)
                
                total_rew += r
                
                if 'metrics' in info:
                    ep_wrs += info['metrics'].get('wrs_penalty', 0)
                    ep_cost += info['metrics'].get('rddl_cost', 0)
                
                state = next_state
                if done: break
            
            rewards.append(total_rew)
            costs.append(ep_cost)
            wrs_scores.append(ep_wrs)
            
            # Decay epsilon for Q-Learner
            if learns:
                agent.epsilon = max(0.05, agent.epsilon * 0.995)

        # End Timer
        total_duration = time.time() - start_clock
        # Training time is 0 for baselines (they don't train)
        train_time = total_duration if learns else 0.0
        
        plot_data[name] = (rewards, color)
        
        # Aggregate Stats
        row = {
            "Agent": name,
            "Avg Reward": np.mean(rewards),
            "Avg Cost": np.mean(costs),
            "Avg WRS Penalty": np.mean(wrs_scores),
            "Avg Plan Time (ms)": np.mean(planning_times) * 1000,
            "Total Train Time (s)": train_time
        }
        results_table.append(row)

    # 5. Output Results
    df = pd.DataFrame(results_table)
    print("\n=== REAL WORLD SENTIMENT RESULTS ===")
    
    formatters = {
        'Avg Reward': '{:.2f}'.format,
        'Avg Cost': '{:.2f}'.format,
        'Avg WRS Penalty': '{:.2f}'.format,
        'Avg Plan Time (ms)': '{:.3f}'.format,
        'Total Train Time (s)': '{:.2f}'.format
    }
    print(df.to_string(index=False, formatters=formatters))
    
    df.to_csv("sentiment_results_table.csv", index=False)
    
    print("\n=== LaTeX Table Code ===")
    print(df.round(3).to_latex(index=False, caption="Performance on Real-World Sentiment Task.", label="tab:sentiment_results"))

    # 6. Plot
    plt.figure(figsize=(12, 6))
    for name, (rew, col) in plot_data.items():
        window = 20
        smoothed = np.convolve(rew, np.ones(window)/window, mode='valid')
        lw = 2.5 if "Q-Learning" in name else 1.5
        plt.plot(smoothed, label=name, color=col, linewidth=lw)
    
    plt.title("Sentiment Pipeline Optimization (Real World Data)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("sentiment_comparison_plot.png")
    print("\nSaved plot to 'sentiment_comparison_plot.png'")

if __name__ == "__main__":
    run_experiment()