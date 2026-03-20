import pyRDDLGym
import os
import random
from tqdm import trange # For a nice training progress bar
import numpy as np
import matplotlib.pyplot as plt

# THE Q-PLANNER CLASS

class PipelineQPlanner:
    def __init__(self, action_space, stage_map, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = {} 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = action_space
        self.stage_map = stage_map 

    def get_current_stage(self, state):
        for key, val in state.items():
            if "current_stage" in key and val == True:
                return key.split("___")[-1] 
        return None

    def get_state_key(self, state):
        """Creates a unique Q-table key based on (Stage, Last_Family_Used)."""
        current_stage = self.get_current_stage(state)
        if not current_stage:
            return "DONE" # Terminal State

        last_family = 'None'
        for key, val in state.items():
            if "last_family_used" in key and val == True:
                last_family = key.split("___")[-1]
                break
        
        return f"{current_stage}__{last_family}"

    def sample_action(self, state, exploration_enabled=True):
        state_key = self.get_state_key(state)
        current_stage = self.get_current_stage(state)
        
        if state_key == "DONE" or not current_stage or current_stage not in self.stage_map:
            return {}

        valid_models = self.stage_map[current_stage]
        
        # Initialize Memory using the combined key
        if state_key not in self.q_table:
            self.q_table[state_key] = {m: 0.0 for m in valid_models}

        # Exploration (Random Choice)
        if exploration_enabled and random.uniform(0, 1) < self.epsilon:
            chosen_model = random.choice(valid_models)
        
        # Exploitation (Max Q-Value)
        else:
            qs = self.q_table[state_key]
            # Handle ties by choosing randomly among tied actions
            max_q = max(qs.values())
            best_actions = [m for m, q in qs.items() if q == max_q]
            chosen_model = random.choice(best_actions)

        return {f"select_model___{chosen_model}": 1}

    def update(self, state, action, reward, next_state):
        curr_state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        used_model = None
        for k, v in action.items():
            if v == 1 and "select_model" in k:
                used_model = k.split("___")[-1]
                break
        
        if curr_state_key not in self.q_table:
            # Should not happen often if sample_action is run first
            stage = self.get_current_stage(state)
            self.q_table[curr_state_key] = {m: 0.0 for m in self.stage_map[stage]}

        if curr_state_key and used_model:
            old_q = self.q_table[curr_state_key][used_model]
            
            # Estimate Future Value (Max Q of the next state)
            next_max_q = 0.0
            if next_state_key != "DONE" and next_state_key in self.q_table:
                # Need to initialize if we land in a new state
                if next_state_key not in self.q_table:
                    stage = self.get_current_stage(next_state)
                    if stage:
                        self.q_table[next_state_key] = {m: 0.0 for m in self.stage_map[stage]}
                
                next_max_q = max(self.q_table[next_state_key].values())
            
            # Bellman Update Rule
            target = reward + self.gamma * next_max_q
            new_q = old_q + self.alpha * (target - old_q)
            
            self.q_table[curr_state_key][used_model] = new_q


class RandomPlanner:
    def __init__(self, action_space, stage_map):
        self.stage_map = stage_map
    
    def get_current_stage(self, state):
        for key, val in state.items():
            if "current_stage" in key and val == True:
                return key.split("___")[-1] 
        return None

    def sample_action(self, state):
        current_stage = self.get_current_stage(state)
        if not current_stage or current_stage not in self.stage_map:
            return {}
        
        valid_models = self.stage_map[current_stage]
        chosen_model = random.choice(valid_models)
        return {f"select_model___{chosen_model}": 1}


# TRAINING AND EVALUATION FUNCTIONS

def train_agent(env, planner, num_episodes):
    print(f"--- Starting Training ({num_episodes} episodes) ---")
    episode_rewards = []
    
    for episode in trange(num_episodes):
        state,  _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = planner.sample_action(state, exploration_enabled=True)
            if not action: break # Safety break if terminal or invalid state
            
            next_state, reward, done, _, info = env.step(action)
            planner.update(state, action, reward, next_state)
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
    
    print("\n--- Training Complete ---")
    return episode_rewards

def evaluate_policy(env, planner, num_episodes, exploration_enabled):
    rewards = []
    for _ in range(num_episodes):
        state, _  = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Random planner doesn't take the exploration_enabled flag, but Q-Planner does
            if isinstance(planner, PipelineQPlanner):
                 action = planner.sample_action(state, exploration_enabled=exploration_enabled)
            else:
                 action = planner.sample_action(state)

            if not action: break
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    return rewards

# MAIN EXECUTION
if __name__ == "__main__":
    # Define Static Domain Knowledge
    STAGE_MODEL_MAP = {
        's1': ['m1_cpu', 'm1_gpu'],
        's2': ['m2_cpu', 'm2_gpu'],
        's3': ['m3_cpu', 'm3_gpu'],
    }

    STAGE_MODEL_MAP = {
    's1': ['m1_cpu', 'm1_gpu'],
    's2': ['m2_cpu', 'm2_gpu'],
    's3': ['m3_cpu', 'm3_gpu'],
    's4': ['m4_cpu', 'm4_gpu'],
    's5': ['m5_cpu', 'm5_gpu'],
    's6': ['m6_cpu', 'm6_gpu'],
    's7': ['m7_cpu', 'm7_gpu'],
    's8': ['m8_cpu', 'm8_gpu'],
    's9': ['m9_cpu', 'm9_gpu'],
    's10': ['m10_cpu', 'm10_gpu'],
    }   

    # Initialize Environment
    env = pyRDDLGym.make("domains/chain_with_model_family.rddl", "instances/chain_with_model_family_problem2.rddl")
    

    # Initialize Agents
    Q_PLANNER = PipelineQPlanner(
        action_space=env.action_space, 
        stage_map=STAGE_MODEL_MAP,
        alpha=0.2,     # Higher learning rate speeds convergence here
        gamma=0.9,     # Moderate discount factor
        epsilon=0.1    # 10% chance of random action
    )
    RANDOM_PLANNER = RandomPlanner(env.action_space, STAGE_MODEL_MAP)
    
    NUM_TRAINING_EPISODES = 10
    NUM_EVAL_EPISODES = 100
    
    # Train Q-Learning Agent
    q_learning_history = train_agent(env, Q_PLANNER, NUM_TRAINING_EPISODES)
    
    # Evaluation
    # Evaluate the learned Q-policy (exploit only).
    q_eval_rewards = evaluate_policy(env, Q_PLANNER, NUM_EVAL_EPISODES, exploration_enabled=False)
    
    # Evaluate the random baseline
    random_eval_rewards = evaluate_policy(env, RANDOM_PLANNER, NUM_EVAL_EPISODES, exploration_enabled=False) # Exploration is irrelevant for Random
    
    # Calculate Averages
    q_avg_reward = np.mean(q_eval_rewards)
    random_avg_reward = np.mean(random_eval_rewards)
    
    # Plotting the Results
    
    # Plot 1: Training Progress
    plt.figure(figsize=(10, 5))
    plt.plot(q_learning_history, label='Q-Learning Reward', color='tab:blue')
    plt.axhline(y=-33.0, color='g', linestyle='--', label='Optimal Reward (-33.0)')
    plt.axhline(y=-50.0, color='r', linestyle='--', label='Greedy Reward (-50.0)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Higher is Better)')
    plt.title('Q-Learning Agent Training Convergence')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('q_learning_training_convergence.png')
    

    # Evaluation Comparison
    plt.figure(figsize=(8, 6))
    planners = ['Q-Learning (Learned)', 'Random Baseline']
    avg_rewards = [q_avg_reward, random_avg_reward]
    
    plt.bar(planners, avg_rewards, color=['tab:blue', 'tab:orange'])
    plt.axhline(y=-33.0, color='g', linestyle='-', label='Optimal Reward (-33.0)')
    plt.axhline(y=np.mean(random_eval_rewards), color='r', linestyle='--', label=f'Random Avg Reward ({random_avg_reward:.2f})')
    plt.ylabel('Average Total Reward (Higher is Better)')
    plt.title('Performance Comparison: Q-Learning vs. Random Agent')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('q_learning_vs_random_performance.png')
    

    print("\n--- Final Summary ---")
    print(f"Optimal Reward: -33.0 (GPU Lane)")
    print(f"Q-Learning Average Reward: {q_avg_reward:.2f}")
    print(f"Random Agent Average Reward: {random_avg_reward:.2f}")
    