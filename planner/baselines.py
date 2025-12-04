import random
import numpy as np
# We need calc_wrs to estimate fairness on the fly
from env.metric_utils import calc_wrs

class RandomPipelinePlanner:
    """Baseline 1: Random Agent (Noise Floor)."""
    def __init__(self, stage_map):
        self.stage_map = stage_map

    def sample_action(self, state):
        current_stage = None
        for key, val in state.items():
            if "current_stage" in key and val == True:
                current_stage = key.split("___")[-1]
                break
        
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage]
            choice = random.choice(options)
            return {f"select_model___{choice}": 1}
        return {}

    def update(self, state, action, reward, next_state): pass


class FixedPipelinePlanner:
    """Baseline 2: Static Policy (Legacy System)."""
    def __init__(self, stage_map, selection_index=0, name="Fixed"):
        self.stage_map = stage_map
        self.selection_index = selection_index
        self.name = name

    def sample_action(self, state):
        current_stage = None
        for key, val in state.items():
            if "current_stage" in key and val == True:
                current_stage = key.split("___")[-1]
                break
        
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage]
            idx = min(self.selection_index, len(options) - 1)
            choice = options[idx]
            return {f"select_model___{choice}": 1}
        return {}

    def update(self, state, action, reward, next_state): pass


class LookaheadFairnessPlanner:
    """
    Baseline 3: Greedy Fairness Heuristic (k=1 Lookahead).
    
    Strategy:
    - "Myopic Optimization": At stage 'i', evaluate only the immediate options for stage 'i'.
    - Calculate immediate Switching Cost (based on history).
    - Calculate immediate Fairness Penalty (WRS of the column).
    - Pick the model that minimizes (Cost + Penalty) for *this step only*.
    
    Why k=1?
    - Scalable: Works for any pipeline length (10, 50, 100).
    - Realistic: Simulates a system that tries to be fair locally but lacks long-term planning.
    """
    def __init__(self, stage_map, env, name="Heuristic (Fairness Lookahead)"):
        self.stage_map = stage_map
        self.env = env 
        self.name = name

    def _get_active_features(self, state):
        stage = None
        last_fam = None
        for key, val in state.items():
            if val == True:
                if "current_stage" in key:
                    stage = key.split("___")[-1]
                elif "last_used_family" in key:
                    last_fam = key.split("___")[-1]
        return stage, last_fam

    def _get_model_family(self, model_name):
        # Heuristic: In our generator, m1,m3.. are Fam1. m2,m4.. are Fam2.
        try:
            idx = int(model_name.replace('m', '')) - 1
            return "fam_1" if idx % 2 == 0 else "fam_2"
        except:
            return "unknown"

    def sample_action(self, state):
        # 1. Sense Context
        current_stage, last_family = self._get_active_features(state)
        
        if not current_stage or current_stage not in self.stage_map:
            return {}

        options = self.stage_map[current_stage]
        
        best_model = None
        min_cost = float('inf')
        
        # 2. Evaluate Options (Greedy Search k=1)
        for model in options:
            # A. Calculate Switching Cost
            # Base cost (0.5) is constant, so we ignore it for comparison
            switch_cost = 0.0
            this_family = self._get_model_family(model)
            
            if last_family and "none" not in last_family and last_family != "unknown":
                if this_family != last_family:
                    switch_cost = 0.5 # Switching penalty from generator
            
            # B. Calculate Fairness Cost (WRS)
            # Peek at the dataframe column for this model
            col_name = f"{current_stage}_{model}"
            fairness_cost = 0.0
            
            if col_name in self.env.full_df.columns:
                # Use the current sampled batch to estimate fairness
                try:
                    for z in ['Z1', 'Z2', 'Z3']:
                        fairness_cost += calc_wrs(self.env.sampled_df, z, col_name)
                except:
                    pass
            
            # C. Total Immediate Cost
            # Scale WRS by 10.0 (same as Environment Reward function)
            total_score = switch_cost + (fairness_cost * 10.0)
            
            if total_score < min_cost:
                min_cost = total_score
                best_model = model
        
        # Fallback
        if best_model is None:
            best_model = random.choice(options)

        return {f"select_model___{best_model}": 1}

    def update(self, state, action, reward, next_state):
        pass