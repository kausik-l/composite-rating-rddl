import os
import sys
import numpy as np
from env.dynamic_chain_env import DynamicChainRatingEnv
from planner.baselines import FixedPipelinePlanner
from utils.generate_scenario import generate_large_scenario

def debug_episode():
    print("=== STARTING DEBUG SESSION ===")
    
    # 1. Generate a small test case (2 stages, 2 models) to keep logs readable
    root_dir = os.path.dirname(os.path.abspath(__file__))
    N, M = 2, 2
    domain_path, inst_path, csv_path = generate_large_scenario(N, M, output_dir=root_dir)
    
    print(f"\n[1] Loading Environment: {inst_path}")
    env = DynamicChainRatingEnv(domain_path, inst_path, csv_path)
    
    # 2. Inspect the Stage Map (This is likely where the bug is)
    print("\n[2] Inspecting Environment Stage Map:")
    print(f"    Map Keys: {list(env.stage_model_map.keys())}")
    if len(env.stage_model_map) > 0:
        first_key = list(env.stage_model_map.keys())[0]
        print(f"    Example Entry ['{first_key}']: {env.stage_model_map[first_key]}")
    else:
        print("    [ERROR] The Map is EMPTY! The planner will not work.")
    
    # 3. Initialize Agent
    agent = FixedPipelinePlanner(env.stage_model_map, 0)
    print(f"\n[3] Initialized FixedPipelinePlanner (Target: Index 0)")
    
    # 4. Run Episode Step-by-Step
    print("\n[4] Running Episode...")
    state, _ = env.reset()
    
    for step in range(10):
        print(f"\n--- STEP {step+1} ---")
        
        # A. Check State
        # Find the 'True' state keys
        active_states = [k for k, v in state.items() if v == True]
        print(f"    Raw State: {active_states}")
        
        # B. Check Agent Perception
        # Mimic the planner's internal logic
        current_stage = None
        for key in active_states:
            if "current_stage" in key:
                current_stage = key.split("___")[-1]
                break
        print(f"    Planner perceives stage as: '{current_stage}'")
        
        if current_stage in env.stage_model_map:
            print(f"    [OK] Stage '{current_stage}' found in map.")
        else:
            print(f"    [FAIL] Stage '{current_stage}' NOT found in map!")
        
        # C. Check Action
        action = agent.sample_action(state)
        print(f"    Action Selected: {action}")
        
        if not action:
            print("    [WARNING] Agent returned NoOp (Empty Action)")
            
        # D. Execute
        next_state, reward, done, trunc, _ = env.step(action)
        print(f"    Reward: {reward}")
        
        state = next_state
        if done:
            print("\n[DONE] Episode Finished.")
            break

if __name__ == "__main__":
    debug_episode()