import random
import numpy as np
from env.metric_utils import calc_wrs

class RandomPipelinePlanner:
    def __init__(self, stage_map, action_name="select_model"):
        self.stage_map = stage_map
        self.action_name = action_name
    def sample_action(self, state):
        current_stage = self._get_stage(state)
        if current_stage and current_stage in self.stage_map:
            choice = random.choice(self.stage_map[current_stage])
            return {f"{self.action_name}___{choice}": 1}
        return {}
    def _get_stage(self, state):
        for k, v in state.items():
            if "current_stage" in k and v: return k.split("___")[-1]
        return None
    def update(self, s, a, r, ns): pass

class FixedPipelinePlanner:
    def __init__(self, stage_map, selection_index=0, name="Fixed", action_name="select_model"):
        self.stage_map = stage_map
        self.selection_index = selection_index
        self.name = name
        self.action_name = action_name
    def sample_action(self, state):
        current_stage = self._get_stage(state)
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage]
            idx = min(self.selection_index, len(options) - 1)
            choice = options[idx]
            return {f"{self.action_name}___{choice}": 1}
        return {}
    def _get_stage(self, state):
        for k, v in state.items():
            if "current_stage" in k and v: return k.split("___")[-1]
        return None
    def update(self, s, a, r, ns): pass

class LookaheadFairnessPlanner:
    """
    Baseline 3: Greedy Heuristic.
    Optimizes (Cost + WRS) for the current step.
    """
    def __init__(self, stage_map, env, name="Heuristic", action_name="select_model"):
        self.stage_map = stage_map
        self.env = env 
        self.name = name
        self.action_name = action_name

    def _get_context(self, state):
        stage = None; last_fam = None
        for k, v in state.items():
            if v:
                if "current_stage" in k: stage = k.split("___")[-1]
                elif "last_used_family" in k: last_fam = k.split("___")[-1]
        return stage, last_fam

    def sample_action(self, state):
        stage, last_fam = self._get_context(state)
        if not stage or stage not in self.stage_map: return {}
        
        options = self.stage_map[stage]
        best_model = None
        min_total_cost = float('inf')
        
        # Check if we can access data
        has_data = hasattr(self.env, 'sampled_batch') and self.env.sampled_batch is not None
        
        for model in options:
            # 1. Switching Cost Logic (Robust to Domain)
            switch_cost = 0.0
            
            # If we are in the Sentiment Domain (action_name="select_component")
            if self.action_name == "select_component":
                # In Sentiment Task, "trans_danish" and "trans_spanish" are Expensive (0.5)
                # "trans_none" is Cheap (0.0)
                if "trans_danish" in model or "trans_spanish" in model:
                    switch_cost = 0.5
                elif "trans_none" in model:
                    switch_cost = 0.0
                # Models (Stage 2) are cheap (0.1) but here we care about relative cost
                
            # If we are in Synthetic Domain (action_name="select_model")
            else:
                # Fallback to the Odd/Even family logic for Synthetic
                try:
                    clean_name = model.replace('m_', '').replace('m', '')
                    idx = int(clean_name) - 1
                    fam = "fam_1" if idx % 2 == 0 else "fam_2"
                    if last_fam and "none" not in last_fam and fam != last_fam:
                        switch_cost = 0.5
                except: pass

            # 2. Fairness Cost (WRS)
            wrs_cost = 0.0
            if has_data:
                # Construct column name based on domain logic
                col = None
                
                # Sentiment Domain Logic
                if self.action_name == "select_component":
                    # We can only calculate WRS at Stage 2 (Model Selection)
                    # We need to know the current translation to find the column
                    if hasattr(self.env, 'current_translation') and self.env.current_translation:
                        trans = self.env.current_translation
                        clean_model = model.replace('m_', '')
                        col = f"{trans}_{clean_model}"
                
                
                # Calculate if column exists
                if col:
                    df = self.env.sampled_batch if hasattr(self.env, 'sampled_batch') else self.env.sampled_df
                    if col in df.columns:
                        try:
                            for z in ['User_gender']:
                                if z in df.columns:
                                    wrs_cost += calc_wrs(df, z, col)
                        except: pass

            # Total Cost
            total = switch_cost + (wrs_cost * 10.0)
            
            if total < min_total_cost:
                min_total_cost = total
                best_model = model
        
        if best_model is None: best_model = random.choice(options)
        return {f"{self.action_name}___{best_model}": 1}

    def update(self, s, a, r, ns): pass