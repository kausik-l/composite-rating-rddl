import os
import json
import numpy as np
import matplotlib.pyplot as plt
from env.rating_env import ChainRatingEnv
from planner.policy import PipelineQPlanner

# --- Helper to handle NumPy types in JSON ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def run_chain_simulation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    domain_path = os.path.join(current_dir, 'domains', 'chain.rddl')
    instance_path = os.path.join(current_dir, 'instances', 'chain1.rddl')
    data_path = os.path.join(current_dir, 'data', 'input', 'synthetic_chain.csv')
    
    log_file = os.path.join(current_dir, 'simulation_log.json')
    plot_file = os.path.join(current_dir, 'training_results.png')

    env = ChainRatingEnv(domain_path, instance_path, data_path)
    agent = PipelineQPlanner(env.action_space, env.stage_model_map)

    print(f"Training Planner on {env.model.instance_name}...")
    
    num_episodes = 50 
    history = []
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_log = []
        
        for step in range(env.horizon):
            action = agent.sample_action(state)
            
            # Clean State: Convert numpy bools to Python bools
            clean_state = {
                k.split("___")[-1]: bool(v) 
                for k, v in state.items() if v
            }
            
            clean_action = list(action.keys())[0].split("___")[-1] if action else "NoOp"

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state)
            
            episode_log.append({
                "episode": episode + 1,
                "step": step + 1,
                "state": clean_state,
                "action": clean_action,
                "reward": float(reward),
                "done": bool(done)
            })
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        history.append(episode_log)
        rewards_per_episode.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward:.2f}")

    env.close()

    # Save logs using the custom encoder
    with open(log_file, 'w') as f:
        json.dump(history, f, indent=4, cls=NumpyEncoder)
    print(f"\nDetailed logs saved to {log_file}")
    
    generate_plots(rewards_per_episode, agent.q_table, plot_file)

def generate_plots(rewards, q_table, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Learning Curve
    ax1.plot(rewards, marker='o', color='b', alpha=0.7)
    ax1.set_title('Total Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (Higher is better)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Q-Values
    labels = []
    values = []
    colors = []
    
    for stage, models in q_table.items():
        for model, q_val in models.items():
            labels.append(f"{stage}\n{model}")
            values.append(q_val)
            colors.append('skyblue' if 's1' in stage else 'salmon')

    x_pos = np.arange(len(labels))
    ax2.bar(x_pos, values, color=colors, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_title('Learned Model Preferences')
    ax2.set_ylabel('Q-Value')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualizations saved to {output_path}")
    # plt.show() # Uncomment if you have a display

if __name__ == "__main__":
    run_chain_simulation()