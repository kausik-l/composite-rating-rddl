import os
import numpy as np
import matplotlib.pyplot as plt
from utils.generate_scenario import generate_large_scenario
from env.dynamic_chain_env import DynamicChainRatingEnv
from planner.baselines import RandomPipelinePlanner, FixedPipelinePlanner
from planner.policy import PipelineQPlanner # Import your Q-Planner

def run_baseline_experiment(N, M, episodes=50):
    print(f"\n=== STARTING EXPERIMENT: {N} Stages x {M} Models ===")
    
    # 1. Auto-Generate Scenario
    root_dir = os.path.dirname(os.path.abspath(__file__))
    domain_path, inst_path, csv_path = generate_large_scenario(N, M, output_dir=root_dir)
    
    # 2. Load Env
    env = DynamicChainRatingEnv(domain_path, inst_path, csv_path)
    
    # 3. Define Agents
    # NOTE: We include Q-Learning now. It won't crash!
    agents = [
        ("Q-Learning", PipelineQPlanner(env.action_space, env.stage_model_map)),
        ("Random", RandomPipelinePlanner(env.stage_model_map)),
        ("Fixed (First)", FixedPipelinePlanner(env.stage_model_map, 0)),
        ("Fixed (Middle)", FixedPipelinePlanner(env.stage_model_map, M // 2))
    ]
    
    results = {}
    
    # 4. Run Loop
    for name, agent in agents:
        print(f"  > Running {name}...")
        rewards = []
        
        # Reset agent if it has memory (Q-Learning)
        if hasattr(agent, 'q_table'):
            agent.q_table = {} 

        for ep in range(episodes):
            state, _ = env.reset()
            total_rew = 0
            
            # Run until done
            while True:
                action = agent.sample_action(state)
                next_state, r, done, trunc, _ = env.step(action)
                
                # CRITICAL: Allow Q-Learner to learn
                agent.update(state, action, r, next_state)
                
                total_rew += r
                state = next_state
                if done or trunc:
                    break
            rewards.append(total_rew)
            
            # Progress bar for slow runs
            if ep % 50 == 0:
                print(f"    Ep {ep}/{episodes}...")
                
        results[name] = rewards
    
    env.close()
    return results

def plot_large_scale(results, N, M):
    plt.figure(figsize=(12, 7))
    
    for name, rewards in results.items():
        # Moving average smoothing
        window = 20 # Smoother window for noisy large runs
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        else:
            smoothed = rewards
            
        # Make Q-Learning stand out
        if "Q-Learning" in name:
            plt.plot(smoothed, label=name, linewidth=3, color='blue')
        else:
            plt.plot(smoothed, label=name, linewidth=1.5, alpha=0.8)
        
    plt.title(f"Scalability Test: {N} Stages, {M} Models/Stage")
    plt.ylabel("Total Reward")
    plt.xlabel("Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"scale_test_{N}_{M}_with_Q.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    # RECOMMENDATION: Increase episodes to give Q-learning a chance!
    # 100 Stages needs time to propagate rewards back.
    STAGES = 100
    MODELS = 20
    EPISODES = 1000  # Increased from 50 to 500
    
    data = run_baseline_experiment(STAGES, MODELS, episodes=EPISODES)
    plot_large_scale(data, STAGES, MODELS)