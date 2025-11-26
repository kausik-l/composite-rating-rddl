import pandas as pd
import numpy as np
import re
from pyRDDLGym.core.env import RDDLEnv
from env.metric_utils import calc_wrs

class DynamicChainRatingEnv(RDDLEnv):
    """
    A specialized RDDL Environment that integrates Data Science Fairness Metrics.
    
    Updates:
    - step() now returns detailed metrics in the 'info' dictionary.
    """
    def __init__(self, domain_file, instance_file, data_path):
        super().__init__(domain=domain_file, instance=instance_file)
        
        self.full_df = pd.read_csv(data_path)
        self.sampled_df = None
        self.selected_pipeline_cols = []
        self.stage_model_map = {}
        
        stages, models = self._parse_objects_from_rddl(instance_file)
        for s in stages:
            self.stage_model_map[s] = sorted(models)

        if not self.stage_model_map:
            raise ValueError("Failed to parse stages/models from instance file.")

    def _parse_objects_from_rddl(self, rddl_path):
        """Regex helper to extract object lists from RDDL text."""
        stages, models = [], []
        try:
            with open(rddl_path, 'r') as f:
                content = f.read().replace('\n', ' ')
            
            s_match = re.search(r'stage\s*:\s*\{\s*([^}]+)\s*\}', content)
            if s_match:
                stages = [x.strip() for x in s_match.group(1).split(',') if x.strip()]
                
            m_match = re.search(r'model\s*:\s*\{\s*([^}]+)\s*\}', content)
            if m_match:
                models = [x.strip() for x in m_match.group(1).split(',') if x.strip()]
        except Exception as e:
            print(f"Warning: Manual file parsing failed: {e}")
        return stages, models

    def reset(self):
        obs, _ = super().reset()
        self.sampled_df = self.full_df.sample(n=100).copy()
        self.selected_pipeline_cols = []
        return obs, {}

    def step(self, action):
        # 1. Run RDDL Physics
        obs, rddl_reward, terminated, truncated, info = super().step(action)
        
        done = terminated or truncated
        is_pipeline_done = obs.get("pipeline_done", 0) >= 1
        
        # Initialize info metrics if not present
        if 'metrics' not in info:
            info['metrics'] = {'rddl_reward': 0.0, 'fairness_penalty': 0.0}
        
        # Capture the raw structural reward (Base + Switching)
        info['metrics']['rddl_reward'] = float(rddl_reward)

        # 2. Map to Data
        selected_model_rddl = None
        for k, v in action.items():
            if v == 1 and "select_model" in k:
                selected_model_rddl = k.split("___")[-1]
                break
        
        if selected_model_rddl:
            current_step_idx = len(self.selected_pipeline_cols)
            if current_step_idx < len(self.stage_model_map):
                sorted_stages = sorted(list(self.stage_model_map.keys()), key=lambda x: int(x[1:]))
                current_stage = sorted_stages[current_step_idx]
                col_name = f"{current_stage}_{selected_model_rddl}"
                if col_name in self.full_df.columns:
                    self.selected_pipeline_cols.append(col_name)

        # 3. Fairness Evaluation
        total_reward = rddl_reward
        
        if is_pipeline_done and len(self.selected_pipeline_cols) > 0:
            final_col = self.selected_pipeline_cols[-1]
            total_wrs = 0.0
            try:
                for z in ['Z1', 'Z2', 'Z3']:
                    total_wrs += calc_wrs(self.sampled_df, z, final_col)
            except Exception:
                pass
            
            fairness_penalty = (total_wrs * 10.0)
            total_reward -= fairness_penalty
            
            # Log the specific penalty for analysis
            info['metrics']['fairness_penalty'] = fairness_penalty
            done = True

        return obs, total_reward, done, truncated, info