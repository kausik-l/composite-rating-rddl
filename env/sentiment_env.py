import pandas as pd
import numpy as np
from pyRDDLGym.core.env import RDDLEnv
from env.metric_utils import calc_wrs

class SentimentPipelineEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_path, batch_size=100):
        super().__init__(domain=domain_file, instance=instance_file)
        
        # Load the MERGED Master CSV
        self.full_df = pd.read_csv(data_path)
        self.batch_size = batch_size
        self.sampled_batch = None
        
        # Track choices
        self.current_translation = None 
        self.current_model = None       

    def reset(self):
        obs, _ = super().reset()
        # Sample a random batch of texts for this episode
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
        
        # Track State
        if selected_comp:
            if "trans" in selected_comp:
                if "none" in selected_comp: self.current_translation = "eng"
                elif "danish" in selected_comp: self.current_translation = "dan"
                elif "spanish" in selected_comp: self.current_translation = "spa"
            elif "m_" in selected_comp:
                # Extract 'bf' from 'm_bf'
                self.current_model = selected_comp.replace("m_", "") 

        # Calculate Reward
        wrs_penalty = 0.0
        
        if is_done and self.current_translation and self.current_model:
            # Construct Column Name: e.g., "dan_bf"
            pred_col = f"{self.current_translation}_{self.current_model}"
            
            if pred_col in self.sampled_batch.columns:
                # Protected Attribute is 'User_gender'
                if 'User_gender' in self.sampled_batch.columns:
                    try:
                        # Calculate WRS on this batch
                        # Z=User_gender, Y=pred_col
                        wrs = calc_wrs(self.sampled_batch, 'User_gender', pred_col)
                        
                        # Scale penalty: If WRS is 0.1, Penalty = 1.0
                        # This makes it comparable to Translation Cost (0.5)
                        wrs_penalty = wrs * 10.0 
                    except:
                        pass
            
            done = True

        # Total = (Negative RDDL Cost) - (Fairness Penalty)
        total_reward = rddl_reward - wrs_penalty
        
        if 'metrics' not in info: info['metrics'] = {}
        info['metrics']['wrs_penalty'] = wrs_penalty
        info['metrics']['rddl_cost'] = abs(rddl_reward)

        return obs, total_reward, done, truncated, info