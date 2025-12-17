import os
import numpy as np
import matplotlib.pyplot as plt
from env.rating_env import ChainRatingEnv
from env.metric_utils import calc_wrs
from planner.policy import PipelineQPlanner
from planner.baselines import RandomPipelinePlanner, FixedPipelinePlanner

def get_oracle_wrs(env):
    """Calculates the best possible WRS for the current data batch."""
    df = env.sampled_df
    
    # We only really need to check the final stage columns
    # because the pipeline structure is fixed (s1 -> s2)
    possible_outputs = ['s2_m21', 's2_m22']
    
    best_wrs = float('inf')
    for col in possible_outputs:
        total_wrs = 0.0
        for prot_attr in ['Z1', 'Z2', 'Z3']:
            total_wrs += calc_wrs(df, prot_attr, col)
        if total_wrs < best_wrs:
            best_wrs = total_wrs
            
    return best_wrs

def run_experiment(env, agent, num_episodes=150):
    regrets = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        optimal_wrs = get_oracle_wrs(env)
        
        agent_wrs = 0.0
        
        for step in range(env.horizon):
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state)
            
            # FIX: Correctly extract WRS from the immediate reward
            # Reward = -5.0 - WRS
            # Therefore: WRS = -(Reward + 5.0)
            if done:
                # We use 5.0 because that's the cost in chain.rddl
                # Adding a small epsilon for float stability
                extracted_wrs = -(reward + 5.0)
                
                # Sanity check: WRS cannot be negative
                agent_wrs = max(0.0, extracted_wrs)

            state = next_state
            if done:
                break
        
        # Calculate Regret
        regret = max(0.0, agent_wrs - optimal_wrs)
        regrets.append(regret)
        
    return regrets

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    domain_path = os.path.join(current_dir, 'domains', 'chain.rddl')
    instance_path = os.path.join(current_dir, 'instances', 'chain1.rddl')
    data_path = os.path.join(current_dir, 'data', 'input', 'synthetic_chain.csv')
    
    env = ChainRatingEnv(domain_path, instance_path, data_path)
    
    EPISODES = 150 

    agents = [
        {"name": "Q-Learning", "agent": PipelineQPlanner(env.action_space, env.stage_model_map), "color": "blue", "style": "-"},
        {"name": "Random",     "agent": RandomPipelinePlanner(env.stage_model_map),               "color": "gray", "style": "--"},
        {"name": "Fixed A (m21)", "agent": FixedPipelinePlanner(env.stage_model_map, 0),          "color": "red",  "style": ":"},
        {"name": "Fixed B (m22)", "agent": FixedPipelinePlanner(env.stage_model_map, 1),          "color": "green", "style": ":"}
    ]

    results = {}
    print(f"Running Comparison ({EPISODES} Episodes)...")
    
    for item in agents:
        print(f"Evaluating {item['name']}...")
        regret_history = run_experiment(env, item['agent'], EPISODES)
        results[item['name']] = regret_history

    env.close()
    plot_regret(results)

def plot_regret(results):
    plt.figure(figsize=(10, 6))
    
    for name, regrets in results.items():
        # Smoothing window
        window = 15 
        smoothed = np.convolve(regrets, np.ones(window)/window, mode='valid')
        
        # Style matching
        style = '-'
        color = 'black'
        if 'Q-Learning' in name: color, style = 'blue', '-'
        elif 'Random' in name:   color, style = 'gray', '--'
        elif 'Fixed A' in name:  color, style = 'red', ':'
        elif 'Fixed B' in name:  color, style = 'green', ':'
            
        plt.plot(smoothed, label=name, color=color, linestyle=style, linewidth=2)

    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.title("Agent Regret (Difference from Optimal Choice)")
    plt.xlabel("Episode")
    plt.ylabel("Regret (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regret_plot.png")
    print("Saved 'regret_plot.png'")
    plt.show()

if __name__ == "__main__":
    main()