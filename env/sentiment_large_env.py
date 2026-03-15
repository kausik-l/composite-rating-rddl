import pandas as pd
import numpy as np
from pyRDDLGym.core.env import RDDLEnv
from env.metric_utils import calc_wrs
from utils.causal_metrics import compute_arc_metrics

class SentimentPipelineEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_path, batch_size=100, reward_mode="WRS"):
        super().__init__(domain=domain_file, instance=instance_file)
        
        self.full_df = pd.read_csv(data_path)
        self.batch_size = batch_size
        self.reward_mode = reward_mode
        self.sampled_batch = None
        
        self.current_translation = None 
        self.current_model = None       

    def reset(self):
        obs, _ = super().reset()
        self.sampled_batch = self.full_df.sample(n=self.batch_size).copy()
        self.current_translation = None
        self.current_model = None
        return obs, {}

    def step(self, action):
        obs, rddl_reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        is_done = obs.get("pipeline_done", 0) >= 1
        
        # Identify Action
        selected_comp = None
        for k, v in action.items():
            if v == 1 and "select_component" in k:
                selected_comp = k.split("___")[-1]
                break
        
        if selected_comp:
            if "trans" in selected_comp:
                if "none" in selected_comp: self.current_translation = "eng"
                elif "danish" in selected_comp: self.current_translation = "dan"
                elif "spanish" in selected_comp: self.current_translation = "spa"
            elif "m_" in selected_comp:
                self.current_model = selected_comp.replace("m_", "") 

        fairness_penalty = 0.0
        
        # Initialize logs
        raw_wrs = 0.0
        raw_die = 0.0
        raw_ate = 0.0 # Placeholder
        
        if is_done and self.current_translation and self.current_model:
            pred_col = f"{self.current_translation}_{self.current_model}"
            
            if pred_col in self.sampled_batch.columns:
                
                target_rows = self.sampled_batch
                
                # -------------------------------------------------------------
                # 1. ALWAYS CALCULATE WRS (Safe to run always)
                # -------------------------------------------------------------
                if len(target_rows) > 10 and 'User_gender' in target_rows.columns:
                    try:
                        target_rows = target_rows.copy()
                        target_rows['gender_binary'] = (target_rows['User_gender'] == 2).astype(int)
                        raw_wrs = calc_wrs(target_rows, 'gender_binary', pred_col)
                    except: pass

                # -------------------------------------------------------------
                # 2. CALCULATE DIE (Only if needed for Reward Mode)
                # Optimization: Skip this expensive calculation if we are only training on WRS
                # -------------------------------------------------------------
                if self.reward_mode in ["DIE", "BOTH"]:
                    if len(target_rows) > 10 and 'C_num' in target_rows.columns and 'User_gender' in target_rows.columns:
                        try:
                            metrics = compute_arc_metrics(
                                target_rows, 
                                treatment_col='C_num',       # Treatment = Utterance ID
                                outcome_col=pred_col, 
                                confounders=['User_gender']  # Confounder = Gender
                            )
                            raw_die = abs(metrics['DIE_Confounding'])
                        except: pass

                # -------------------------------------------------------------
                # 3. APPLY REWARD PENALTY (Based on Mode)
                # -------------------------------------------------------------
                if self.reward_mode == "WRS":
                    fairness_penalty += (raw_wrs)

                elif self.reward_mode == "DIE":
                    fairness_penalty += (raw_die * 10.0)

                elif self.reward_mode == "BOTH":
                    fairness_penalty += (raw_wrs)
                    fairness_penalty += (raw_die * 10.0)
            
            done = True

        # Total Reward (ATE is excluded)
        total_reward = rddl_reward - fairness_penalty 
        
        if 'metrics' not in info: info['metrics'] = {}
        info['metrics']['wrs_penalty'] = fairness_penalty
        info['metrics']['rddl_cost'] = abs(rddl_reward)
        
        # Log the raw values so they appear in the final table
        info['metrics']['raw_wrs'] = float(raw_wrs)
        info['metrics']['raw_die'] = float(raw_die)
        info['metrics']['raw_ate'] = 0.0

        return obs, total_reward, done, truncated, info