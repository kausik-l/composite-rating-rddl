import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.generate_scenario import generate_large_scenario
from env.dynamic_chain_env import DynamicChainRatingEnv
from planner.policy import ContextAwareQPlanner
from planner.baselines import RandomPipelinePlanner, FixedPipelinePlanner, BestFirstSearchPlanner

# =============================================================================
# GLOBAL CONFIGURATION
# Change these values here to update ALL experiments at once.
# =============================================================================
CONFIG = {
    # Environment Settings
    "NUM_STAGES": 50,       # Length of the pipeline
    "NUM_MODELS": 20,        # Number of choices per stage
    "NUM_FAMILIES": 2,      # Number of 'Families' 
    
    # Training Settings
    "TRAIN_EPISODES": 1000,  # How long the Q-Learner gets to learn
    
    # Evaluation Settings
    "EVAL_EPISODES": 1000,   # How many episodes to average for the final table
    "HEATMAP_EPISODES": 1000, # How many episodes to plot in the heatmap
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_training_session(env, episodes=CONFIG["TRAIN_EPISODES"]):
    """
    Trains a fresh Context-Aware Q-Learning agent from scratch.
    Returns the trained agent ready for evaluation.
    """
    agent = ContextAwareQPlanner(env.action_space, env.stage_model_map, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    # Use tqdm for a nice progress bar during training
    for _ in tqdm(range(episodes), leave=False, desc="Training Agent"):
        state, _ = env.reset()
        while True:
            action = agent.sample_action(state)
            next_state, r, done, trunc, _ = env.step(action)
            agent.update(state, action, r, next_state)
            if done or trunc: break
    return agent

def get_paths():
    """Helper to get consistent file paths."""
    root = os.path.dirname(os.path.abspath(__file__))
    return root

# =============================================================================
# EXPERIMENT 1: POLICY HEATMAP
# Visualizes the path the agent takes (Lane Learning).
# =============================================================================
def eval_policy_heatmap():
    print(f"\n[1/3] Generating Policy Heatmap ({CONFIG['NUM_STAGES']} Stages)...")
    
    root = get_paths()
    d, i, c = generate_large_scenario(
        CONFIG["NUM_STAGES"], 
        CONFIG["NUM_MODELS"], 
        CONFIG["NUM_FAMILIES"], 
        output_dir=root
    )
    env = DynamicChainRatingEnv(d, i, c)
    
    # 1. Train the agent
    agent = run_training_session(env, episodes=CONFIG["TRAIN_EPISODES"])
    
    # 2. Collect Trajectories (Exploitation Mode)
    selection_counts = np.zeros((CONFIG["NUM_MODELS"], CONFIG["NUM_STAGES"]))
    
    print("  Collecting trajectories...")
    for _ in range(CONFIG["HEATMAP_EPISODES"]):
        state, _ = env.reset()
        step_idx = 0
        agent.epsilon = 0.0 # Turn off randomness to see the learned strategy
        
        while True:
            action = agent.sample_action(state)
            
            # Extract model index from action name "select_model___m3" -> 2
            model_name = list(action.keys())[0].split("___")[-1]
            model_idx = int(model_name.replace('m', '')) - 1 
            
            if step_idx < CONFIG["NUM_STAGES"]:
                selection_counts[model_idx, step_idx] += 1
            
            next_state, _, done, trunc, _ = env.step(action)
            state = next_state
            step_idx += 1
            if done or trunc: break

    # 3. Plot Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(selection_counts, cmap="GnBu", annot=False, cbar_kws={'label': 'Selection Frequency'})
    
    plt.title(f"Agent Strategy Heatmap ({CONFIG['NUM_STAGES']} Stages)")
    plt.xlabel("Pipeline Stage")
    plt.ylabel("Model Choice")
    
    # Label Y-axis nicely (m1, m2...)
    plt.yticks(ticks=np.arange(CONFIG["NUM_MODELS"])+0.5, 
               labels=[f"m{i+1}" for i in range(CONFIG["NUM_MODELS"])], 
               rotation=0)
    
    plt.savefig("eval_policy_heatmap.png")
    print("Saved 'eval_policy_heatmap.png'")
    env.close()

# =============================================================================
# EXPERIMENT 2: COST SENSITIVITY
# Tests if the agent rationally adapts to changing switching costs.
# =============================================================================
def eval_cost_sensitivity():
    print("\n[2/3] Running Cost Sensitivity Analysis...")
    
    costs_to_test = [0.1, 0.5, 1.0, 2.0, 5.0]
    final_rewards = []
    root = get_paths()
    
    for cost in costs_to_test:
        print(f"  Testing Switching Cost: {cost}")
        
        # 1. Generate Scenario
        d, i, c = generate_large_scenario(
            CONFIG["NUM_STAGES"], 
            CONFIG["NUM_MODELS"], 
            CONFIG["NUM_FAMILIES"], 
            output_dir=root
        )
        
        # 2. Patch the RDDL Instance to change the cost dynamically
        # We look for the default value '0.5' and swap it.
        with open(i, 'r') as f: content = f.read()
        content = content.replace("switching_cost = 0.5;", f"switching_cost = {cost};")
        with open(i, 'w') as f: f.write(content)
        
        # 3. Train on this specific cost landscape
        env = DynamicChainRatingEnv(d, i, c)
        agent = run_training_session(env, episodes=500) # Slightly shorter training for speed
        
        # 4. Evaluate Average Utility
        total_score = 0
        for _ in range(50):
            state, _ = env.reset()
            while True:
                action = agent.sample_action(state)
                next_state, r, done, trunc, _ = env.step(action)
                total_score += r
                state = next_state
                if done: break
        
        final_rewards.append(total_score / 50.0)
        env.close()

    # 5. Plot Sensitivity Curve
    plt.figure(figsize=(8, 5))
    plt.plot(costs_to_test, final_rewards, marker='o', color='crimson', linewidth=2)
    plt.title("Sensitivity Analysis: Impact of Switching Cost")
    plt.xlabel("Switching Cost Penalty")
    plt.ylabel("Average Total Reward")
    plt.grid(True, alpha=0.3)
    plt.savefig("eval_cost_sensitivity.png")
    print("Saved 'eval_cost_sensitivity.png'")

# =============================================================================
# EXPERIMENT 3: QUANTITATIVE TABLE (LaTeX)
# Generates the detailed breakdown for your paper.
# =============================================================================
def eval_paper_table():
    print("\n[3/3] Generating Quantitative Table...")
    
    root = get_paths()
    d, i, c = generate_large_scenario(
        CONFIG["NUM_STAGES"], 
        CONFIG["NUM_MODELS"], 
        CONFIG["NUM_FAMILIES"], 
        output_dir=root
    )
    env = DynamicChainRatingEnv(d, i, c)
    
    # Calculate fixed structural cost for decomposition
    # Base cost in generator is 0.5 per stage
    base_pipeline_cost = CONFIG["NUM_STAGES"] * 0.5
    
    # Train Q-Learning Agent first
    q_agent = run_training_session(env, episodes=CONFIG["TRAIN_EPISODES"])
    q_agent.epsilon = 0.0 # Evaluation mode
    
    # Define all agents to compare
    agents = [
        ("Context-Aware Q-Learning", q_agent),
        ("Heuristic (Best First)", BestFirstSearchPlanner(env.stage_model_map)),
        ("Fixed (Biased Family)", FixedPipelinePlanner(env.stage_model_map, 0)),
        ("Fixed (Fair Family)", FixedPipelinePlanner(env.stage_model_map, 1)),
        ("Random", RandomPipelinePlanner(env.stage_model_map))
    ]
    
    results = []
    
    # Run Evaluation Loop
    for name, agent in agents:
        metrics = {'Fairness Penalty': [], 'Switching Cost': [], 'Total Reward': []}
        
        for _ in range(CONFIG["EVAL_EPISODES"]):
            state, _ = env.reset()
            ep_rddl_reward = 0
            ep_fairness = 0
            
            while True:
                action = agent.sample_action(state)
                next_state, total_r, done, trunc, info = env.step(action)
                
                # Extract RDDL Reward (Base + Switching) and Fairness (WRS)
                if 'metrics' in info:
                    ep_rddl_reward += info['metrics'].get('rddl_reward', 0)
                    if info['metrics'].get('fairness_penalty', 0) > 0:
                        ep_fairness = info['metrics']['fairness_penalty']
                
                state = next_state
                if done or trunc: break
            
            # Calculate Switching Cost: -(RDDL_Reward + Fixed_Base_Cost)
            # We clamp at 0 to avoid floating point artifacts
            switching_cost = max(0.0, -(ep_rddl_reward + base_pipeline_cost))
            
            metrics['Fairness Penalty'].append(ep_fairness)
            metrics['Switching Cost'].append(switching_cost)
            metrics['Total Reward'].append(total_r)
            
        # Compute Means
        row = {
            "Agent": name,
            "Fairness Penalty": np.mean(metrics['Fairness Penalty']),
            "Switching Cost": np.mean(metrics['Switching Cost']),
            "Total Reward": np.mean(metrics['Total Reward'])
        }
        results.append(row)

    # Export
    df = pd.DataFrame(results)
    print("\n=== EXPERIMENT RESULTS ===")
    print(df.round(3).to_string(index=False))
    
    print("\n=== LaTeX Table Code ===")
    caption = f"Performance comparison on {CONFIG['NUM_STAGES']}-stage pipeline. Lower penalties/costs are better. Higher Total Reward is better."
    print(df.round(2).to_latex(index=False, caption=caption, label="tab:results"))
    
    df.to_csv("experiment_results_table.csv", index=False)
    env.close()

if __name__ == "__main__":
    # Run all evaluations using the Global Config
    eval_policy_heatmap()
    eval_cost_sensitivity()
    eval_paper_table()