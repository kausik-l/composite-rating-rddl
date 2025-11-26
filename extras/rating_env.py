import numpy as np
import pandas as pd
import pyRDDLGym
from pyRDDLGym.core.env import RDDLEnv
from env.metric_utils import calc_wrs

class ChainRatingEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_path):
        super().__init__(domain=domain_file, instance=instance_file)
        
        self.full_df = pd.read_csv(data_path)
        self.sampled_df = None
        
        self.stage_model_map = {
            's1': ['m11', 'm12'],
            's2': ['m21', 'm22']
        }
        self.model_to_csv_col = {
            'm11': 's1_m11', 'm12': 's1_m12',
            'm21': 's2_m21', 'm22': 's2_m22'
        }
        self.selected_pipeline_cols = []

    def reset(self):
        obs, _ = super().reset() 
        self.sampled_df = self.full_df.sample(n=100).copy()
        self.selected_pipeline_cols = []
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check if RDDL says pipeline is done
        # We check if 'pipeline_done' is True (or 1) in the observation
        is_done = obs.get("pipeline_done", 0) == 1
        
        # Identify selected model
        selected_model = None
        for action_name, val in action.items():
            if val == 1 and "select_model" in action_name:
                selected_model = action_name.split("___")[-1]
                break
        


        # Calculate WRS at every step, not just when done
        if selected_model and selected_model in self.model_to_csv_col:
            col_name = self.model_to_csv_col[selected_model]
            self.selected_pipeline_cols.append(col_name)
            
            # Calculate WRS immediately for this specific model's output
            current_step_wrs = 0.0
            for prot_attr in ['Z1', 'Z2', 'Z3']:
                current_step_wrs += calc_wrs(self.sampled_df, prot_attr, col_name)
            
            # Apply penalty immediately
            reward -= current_step_wrs
        
        # Apply WRS only when finishing
        if is_done and len(self.selected_pipeline_cols) > 0:
            final_output_col = self.selected_pipeline_cols[-1]
            total_wrs = 0.0
            for prot_attr in ['Z1', 'Z2', 'Z3']:
                wrs = calc_wrs(self.sampled_df, prot_attr, final_output_col)
                total_wrs += wrs
            reward -= total_wrs

        # FIX: Force termination if pipeline is done
        done = terminated or truncated or is_done
        
        return obs, reward, done, False, info