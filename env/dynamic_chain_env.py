import pandas as pd
import numpy as np
import re
from pyRDDLGym.core.env import RDDLEnv
from env.metric_utils import calc_wrs
from utils.causal_metrics import compute_arc_metrics

class DynamicChainRatingEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_path, reward_mode="WRS"):
        super().__init__(domain=domain_file, instance=instance_file)
        
        self.full_df = pd.read_csv(data_path)
        self.sampled_df = None
        self.selected_pipeline_cols = []
        self.stage_model_map = {}
        self.reward_mode = reward_mode
        
        stages, models = self._parse_objects_from_rddl(instance_file)
        for s in stages:
            self.stage_model_map[s] = sorted(models)

        if not self.stage_model_map:
            raise ValueError("Failed to parse stages/models from instance file.")

    def _parse_objects_from_rddl(self, rddl_path):
        stages, models = [], []
        try:
            with open(rddl_path, 'r') as f:
                content = f.read().replace('\n', ' ')
            s_match = re.search(r'stage\s*:\s*\{\s*([^}]+)\s*\}', content)
            if s_match: stages = [x.strip() for x in s_match.group(1).split(',') if x.strip()]
            m_match = re.search(r'model\s*:\s*\{\s*([^}]+)\s*\}', content)
            if m_match: models = [x.strip() for x in m_match.group(1).split(',') if x.strip()]
        except Exception as e:
            print(f"Warning: Manual file parsing failed: {e}")
        return stages, models

    def reset(self):
        obs, _ = super().reset()
        self.sampled_df = self.full_df.sample(n=500).copy()
        self.selected_pipeline_cols = []
        return obs, {}

    def step(self, action):
        obs, rddl_reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        
        if 'metrics' not in info:
            info['metrics'] = {'rddl_reward': 0.0, 'fairness_penalty': 0.0, 'merit_reward': 0.0}
        
        selected_model_rddl = None
        for k, v in action.items():
            if v == 1 and "select_model" in k:
                selected_model_rddl = k.split("___")[-1]
                break
        
        fairness_penalty = 0.0
        merit_reward = 0.0
        
        if selected_model_rddl:
            current_step_idx = len(self.selected_pipeline_cols)
            if current_step_idx < len(self.stage_model_map):
                sorted_stages = sorted(list(self.stage_model_map.keys()), key=lambda x: int(x[1:]))
                current_stage = sorted_stages[current_step_idx]
                col_name = f"{current_stage}_{selected_model_rddl}"
                
                if col_name in self.full_df.columns:
                    self.selected_pipeline_cols.append(col_name)
                    
                    try:
                        # WRS (Statistical Bias) -> Penalty
                        if self.reward_mode in ["WRS", "BOTH"]:
                            wrs_val = sum(calc_wrs(self.sampled_df, z, col_name) for z in ['Z1','Z2','Z3'])
                            fairness_penalty += (wrs_val * 10.0)

                        # DIE (Causal Metrics)
                        if self.reward_mode in ["DIE", "BOTH"]:
                            # Calculate Merit & Confounding
                            metrics = compute_arc_metrics(self.sampled_df, treatment_col='T', outcome_col=col_name, confounders=['Z1'])
                            
                            # Reward: True Merit (ATE)
                            merit_reward += (metrics['ATE_Merit'] * 10.0)
                            
                            # Penalty: Confounding Impact (DIE)
                            fairness_penalty += (metrics['DIE_Confounding'] * 20.0)

                    except:
                        pass

        # Final Reward = RDDL_Cost - Fairness_Penalty + Merit_Reward
        total_reward = rddl_reward - fairness_penalty + merit_reward
        
        info['metrics']['rddl_reward'] = float(rddl_reward)
        info['metrics']['fairness_penalty'] = float(fairness_penalty)
        info['metrics']['merit_reward'] = float(merit_reward)

        return obs, total_reward, done, truncated, info