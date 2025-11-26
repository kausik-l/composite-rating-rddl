import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.generate_scenario import generate_large_scenario
from env.dynamic_chain_env import DynamicChainRatingEnv
from planner.policy import ContextAwareQPlanner
from planner.baselines import RandomPipelinePlanner, FixedPipelinePlanner, BestFirstSearchPlanner

def run_multi_agent_experiment(num_stages=20, num_models=6, num_families=2, episodes=500):
    print(f"\n=== STARTING MAIN PERFORMANCE EXPERIMENT ===")
    
    # 1. Auto-Generate Environment (Data + Rules)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    domain_path, inst_path, data_path = generate_large_scenario(
        num_stages=num_stages, 
        num_models_per_stage=num_models, 
        num_families=num_families,
        output_dir=root_dir
    )

    env = DynamicChainRatingEnv(domain_path, inst_path, data_path)
    
    # Calculate unavoidable base cost for plotting offsets
    structural_cost = num_stages * 0.5 
    
    # 2. Define Agents
    # Note: Fixed Agent 0 = Biased Family. Fixed Agent 1 = Fair Family.
    agents = [
        {"name": "Context-Aware Q", "agent": ContextAwareQPlanner(env.action_space, env.stage_model_map), "color": "purple", "learns": True},
        {"name": "Heuristic (Best First)", "agent": BestFirstSearchPlanner(env.stage_model_map), "color": "green", "learns": False},
        {"name": "Fixed (Biased Family)", "agent": FixedPipelinePlanner(env.stage_model_map, 0), "color": "red", "learns": False},
        {"name": "Fixed (Fair Family)", "agent": FixedPipelinePlanner(env.stage_model_map, 1), "color": "orange", "learns": False},
        {"name": "Random", "agent": RandomPipelinePlanner(env.stage_model_map), "color": "gray", "learns": False}
    ]
    
    results = {}

    # 3. Run Experiment Loop
    for entry in agents:
        name = entry['name']
        agent = entry['agent']
        print(f"Running {name}...")
        
        rewards = []
        # Reset agent memory if needed
        if entry['learns'] and hasattr(agent, 'q_table'): agent.q_table = {}

        for ep in tqdm(range(episodes)):
            state, _ = env.reset()
            total_rew = 0
            while True:
                action = agent.sample_action(state)
                next_state, r, done, trunc, _ = env.step(action)
                
                if entry['learns']: 
                    agent.update(state, action, r, next_state)
                
                total_rew += r
                state = next_state
                if done or trunc: break
            rewards.append(total_rew)
        results[name] = rewards

    env.close()

    # --- PLOT 1: MACRO VIEW (Total Reward) ---
    plt.figure(figsize=(12, 6))
    for name, rewards in results.items():
        # Apply smoothing for readability
        window = 15
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid') if len(rewards) > window else rewards
        entry = next(item for item in agents if item["name"] == name)
        plt.plot(smoothed, linewidth=2, color=entry['color'], label=name)

    plt.title(f"Performance Comparison: Total Utility ({num_stages} Stages)")
    plt.ylabel("Total Reward (Base Cost + Switching Penalty + Fairness)")
    plt.xlabel("Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("performance_comparison.png")
    print("Saved performance_comparison.png")

    # --- PLOT 2: MICRO VIEW (Fairness Only) ---
    # We subtract the structural costs to isolate the Fairness Penalty.
    plt.figure(figsize=(12, 6))
    for name, rewards in results.items():
        if name == "Random": continue # Skip random as it distorts the scale
        
        # Approx Fairness = -(Reward + Structural_Cost)
        fairness = [-(r + structural_cost) for r in rewards]
        
        window = 20
        smoothed = np.convolve(fairness, np.ones(window)/window, mode='valid') if len(fairness) > window else fairness
        entry = next(item for item in agents if item["name"] == name)
        plt.plot(smoothed, linewidth=2, color=entry['color'], label=name)

    plt.title("Fairness Penalty Analysis (Lower is Better)")
    plt.ylabel("Weighted Rejection Score (Scaled)")
    plt.xlabel("Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("fairness_penalty_detail.png")
    print("Saved fairness_penalty_detail.png")

if __name__ == "__main__":
    run_multi_agent_experiment(num_stages=50, num_models=20, num_families=2, episodes=1000)