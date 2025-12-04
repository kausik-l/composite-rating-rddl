import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import deque

from utils.generate_scenario import generate_large_scenario
from env.dynamic_chain_env import DynamicChainRatingEnv
from planner.policy import ContextAwareQPlanner
from planner.baselines import RandomPipelinePlanner, FixedPipelinePlanner, LookaheadFairnessPlanner
from utils.causal_metrics import compute_arc_metrics
from env.metric_utils import calc_wrs 

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "NUM_STAGES": 10,       
    "NUM_MODELS": 5,         
    "NUM_FAMILIES": 2,       
    "TRAIN_EPISODES": 1000, 
    "EVAL_EPISODES": 500,    
    "HEATMAP_EPISODES": 50,
    "DATA_SAMPLES": 30000,
    "SKIP_TRAINING": False 
}

def get_paths(): return os.path.dirname(os.path.abspath(__file__))

def load_or_train_agent(env, mode_name="WRS"):
    save_dir = os.path.join(get_paths(), "saved_agents")
    save_path = os.path.join(save_dir, f"q_agent_{mode_name}_{CONFIG['NUM_STAGES']}_stages.pkl")
    agent = ContextAwareQPlanner(env.action_space, env.stage_model_map, alpha=0.2, gamma=0.5, epsilon=1.0)
    
    # Try Loading
    if agent.load_agent(save_path):
        print(f"  > [LOADED] {mode_name} Agent found on disk.")
        return agent, 0.0

    # Check Skip Flag
    if CONFIG["SKIP_TRAINING"]:
        print(f"  > [WARNING] {mode_name} Agent missing. Skipping.")
        return None, 0.0

    # Train
    print(f"\n  >>> STARTING TRAINING: Q-Agent ({mode_name}) <<<")
    print(f"  > Environment Reward Mode: {env.reward_mode}")
    print(f"  > Target Episodes: {CONFIG['TRAIN_EPISODES']}")
    
    decay = 0.998
    min_epsilon = 0.05
    start = time.time()
    
    # Reset history
    agent.training_history = []
    # For smoothing the progress bar display
    recent_scores = deque(maxlen=50) 
    
    # TQDM Progress Bar with Stats
    pbar = tqdm(range(CONFIG["TRAIN_EPISODES"]), desc=f"Training {mode_name}", unit="ep")
    
    for _ in pbar:
        state, _ = env.reset()
        ep_total = 0
        while True:
            action = agent.sample_action(state)
            next_state, r, done, trunc, _ = env.step(action)
            agent.update(state, action, r, next_state)
            ep_total += r
            state = next_state
            if done or trunc: break
        
        agent.epsilon = max(min_epsilon, agent.epsilon * decay)
        agent.training_history.append(ep_total)
        recent_scores.append(ep_total)
        
        # Update Progress Bar Description with more Stats
        pbar.set_postfix({
            'Epsilon': f"{agent.epsilon:.2f}",
            'AvgRew(50)': f"{np.mean(recent_scores):.1f}"
        })
    
    duration = time.time() - start
    os.makedirs(save_dir, exist_ok=True)
    agent.save_agent(save_path)
    print(f"  > Training Finished in {duration:.2f}s.\n")
    
    return agent, duration

def run_ablation_study():
    print(f"\n[PHASE 1] Loading Agents...")
    
    root = get_paths()
    d, i, c = generate_large_scenario(CONFIG["NUM_STAGES"], CONFIG["NUM_MODELS"], CONFIG["NUM_FAMILIES"], num_samples=CONFIG["DATA_SAMPLES"], output_dir=root)
    base_cost = CONFIG["NUM_STAGES"] * 0.5 

    # Load/Train Q-Agents
    # We create specific envs to ensure they train on the right path.
    q_agents = {}
    
    # WRS Agent
    print("\n[Agent 1/3] Checking WRS Agent...")
    env_wrs = DynamicChainRatingEnv(d, i, c, reward_mode="WRS")
    agent_wrs, t_wrs = load_or_train_agent(env_wrs, "WRS")
    if agent_wrs: 
        agent_wrs.epsilon = 0.0
        q_agents["WRS"] = (agent_wrs, t_wrs)
    env_wrs.close()

    # DIE Agent
    print("\n[Agent 2/3] Checking DIE Agent...")
    env_die = DynamicChainRatingEnv(d, i, c, reward_mode="DIE")
    agent_die, t_die = load_or_train_agent(env_die, "DIE")
    if agent_die: 
        agent_die.epsilon = 0.0
        q_agents["DIE"] = (agent_die, t_die)
    env_die.close()

    # BOTH Agent
    print("\n[Agent 3/3] Checking BOTH Agent...")
    env_both = DynamicChainRatingEnv(d, i, c, reward_mode="BOTH")
    agent_both, t_both = load_or_train_agent(env_both, "BOTH")
    if agent_both: 
        agent_both.epsilon = 0.0
        q_agents["BOTH"] = (agent_both, t_both)
    env_both.close()


    # 2. Evaluation Phase
    print(f"\n[PHASE 2] Running Evaluation...")
    env_eval = DynamicChainRatingEnv(d, i, c, reward_mode="BOTH") 

    agents = []
    if "WRS" in q_agents: agents.append(("Q-Learning (WRS Only)", q_agents["WRS"][0], q_agents["WRS"][1]))
    if "DIE" in q_agents: agents.append(("Q-Learning (DIE Only)", q_agents["DIE"][0], q_agents["DIE"][1]))
    if "BOTH" in q_agents: agents.append(("Q-Learning (Combined)", q_agents["BOTH"][0], q_agents["BOTH"][1]))
    
    agents.extend([
        ("Heuristic (Lookahead)", LookaheadFairnessPlanner(env_eval.stage_model_map, env_eval), 0.0),
        ("Fixed (Biased)", FixedPipelinePlanner(env_eval.stage_model_map, 0), 0.0),
        ("Fixed (Fair)", FixedPipelinePlanner(env_eval.stage_model_map, 1), 0.0),
        ("Random", RandomPipelinePlanner(env_eval.stage_model_map), 0.0)
    ])
    
    table_data = []
    plot_data = {name: [] for name, _, _ in agents}

    for name, agent, train_time in agents:
        print(f"  > Agent: {name}")
        m_switch, m_total, m_time = [], [], []
        causal_df = None
        
        # Evaluation Progress Bar
        eval_pbar = tqdm(range(CONFIG["EVAL_EPISODES"]), desc=f"    Eval {name}", leave=False)
        
        for ep_idx in eval_pbar:
            state, _ = env_eval.reset()
            ep_total, ep_rddl = 0, 0
            
            while True:
                t0 = time.time()
                action = agent.sample_action(state)
                t1 = time.time()
                m_time.append(t1 - t0)
                
                next_state, r, done, trunc, info = env_eval.step(action)
                if 'metrics' in info: ep_rddl += info['metrics'].get('rddl_reward', 0)
                ep_total += r
                state = next_state
                if done or trunc: break
            
            m_switch.append(max(0.0, -(ep_rddl + base_cost)))
            m_total.append(ep_total)
            
            # Capture last episode for Causal Metrics
            if ep_idx == 0:
                causal_df = env_eval.sampled_df.copy()
                if len(env_eval.selected_pipeline_cols) > 0:
                    final_col = env_eval.selected_pipeline_cols[-1]
                    causal_df['Y_Final'] = env_eval.full_df[final_col].values[env_eval.sampled_df.index]
            
            plot_data[name].append(ep_total)

        # Calculate Metrics
        ate_merit, die_confounding, final_wrs = 0.0, 0.0, 0.0
        
        if causal_df is not None and 'Y_Final' in causal_df.columns:
            # ARC Framework Metrics
            arc_metrics = compute_arc_metrics(causal_df, treatment_col='T', outcome_col='Y_Final', confounders=['Z1'])
            ate_merit = arc_metrics['ATE_Merit']
            die_confounding = arc_metrics['DIE_Confounding']
            
            for z in ['Z1', 'Z2', 'Z3']: final_wrs += calc_wrs(causal_df, z, 'Y_Final')

        N = CONFIG["NUM_STAGES"]
        row = {
            "Agent": name,
            "Total Reward": np.mean(m_total) / N,
            "Switch Cost": np.mean(m_switch) / N,
            "WRS": final_wrs,
            "ATE (Merit)": ate_merit,         
            "DIE (Confounding)": die_confounding, 
            "Time (ms)": np.mean(m_time) * 1000
        }
        table_data.append(row)

    env_eval.close()
    
    # Print and Save Results
    df = pd.DataFrame(table_data)
    print("\n=== FINAL RESULTS ===")
    print(df.to_string(index=False))
    print("\n=== LaTeX Code ===")
    print(df.round(4).to_latex(index=False, caption="Results.", label="tab:res"))
    df.to_csv("final_results_ablation.csv", index=False)
    
    # Plot Evaluation
    plt.figure(figsize=(12, 6))
    for name, rewards in plot_data.items():
        win = 10
        smooth = np.convolve(rewards, np.ones(win)/win, mode='valid')
        ls = '-' if "Q-Learning" in name else '--'
        lw = 2.5 if "Q-Learning" in name else 1.5
        plt.plot(smooth, linewidth=lw, linestyle=ls, label=name)
    plt.legend(); plt.savefig("plot_eval_comparison.png")

    # Plot Training Curves
    plt.figure(figsize=(12, 6))
    has_data = False
    win = 50
    for mode in ["WRS", "DIE", "BOTH"]:
        if mode in q_agents:
            agent = q_agents[mode][0]
            if hasattr(agent, 'training_history') and len(agent.training_history) > 0:
                hist = agent.training_history
                smooth = np.convolve(hist, np.ones(win)/win, mode='valid')
                plt.plot(smooth, label=f"Q-Learning ({mode})", linewidth=2)
                has_data = True
    if has_data:
        plt.title("Learning Efficiency")
        plt.legend()
        plt.savefig("plot_ablation_learning.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train: CONFIG["SKIP_TRAINING"] = False
    run_ablation_study()