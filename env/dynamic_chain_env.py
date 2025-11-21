import pandas as pd
import numpy as np
import re
from pyRDDLGym.core.env import RDDLEnv
from env.metric_utils import calc_wrs

class DynamicChainRatingEnv(RDDLEnv):
    """    
    It extends the standard RDDLEnv to add:
    Automatic parsing of the pipeline structure from RDDL files.
    Integration with a Pandas DataFrame to simulate data processing.
    Calculation of Rewards (WRS) based on the selected path.
    """
    def __init__(self, domain_file, instance_file, data_path):
        # Initialize the standard RDDL simulator
        super().__init__(domain=domain_file, instance=instance_file)
        
        # Load the synthetic data
        self.full_df = pd.read_csv(data_path)
        self.sampled_df = None
        self.selected_pipeline_cols = []
        
        # This dictionary will hold the structure: {'s1': ['m1', 'm2'], 's2': ...}
        self.stage_model_map = {}
        
        # We need to know which models belong to which stage.
        # Because pyRDDLGym internal structures vary by version, we parse 
        # the RDDL text file directly to get the list of objects.
        
        # Extract object names (e.g., ['s1', 's2'] and ['m1', 'm2'])
        stages, models = self._parse_objects_from_rddl(instance_file)
        
        # Build the Map
        # In our experiments, every model is valid at every stage.
        # So we build a full map: s1 -> [m1...m20], s2 -> [m1...m20]
        for s in stages:
            self.stage_model_map[s] = sorted(models)

        if not self.stage_model_map:
            raise ValueError("Failed to parse stages/models from instance file.")

    def _parse_objects_from_rddl(self, rddl_path):
        """
        Helper function to read the .rddl file text and extract object lists using Regex.
        """
        stages = []
        models = []
        try:
            with open(rddl_path, 'r') as f:
                content = f.read()
            
            # Collapse newlines to make regex matching easier
            content = content.replace('\n', ' ')
            
            # Find the block: stage : { s1, s2, ... };
            stage_match = re.search(r'stage\s*:\s*\{\s*([^}]+)\s*\}', content)
            if stage_match:
                raw = stage_match.group(1)
                # Clean up whitespace and split by comma
                stages = [x.strip() for x in raw.split(',') if x.strip()]
                
            # Find the block: model : { m1, m2, ... };
            model_match = re.search(r'model\s*:\s*\{\s*([^}]+)\s*\}', content)
            if model_match:
                raw = model_match.group(1)
                models = [x.strip() for x in raw.split(',') if x.strip()]
                
        except Exception as e:
            print(f"Warning: Manual file parsing failed: {e}")
            
        return stages, models

    def reset(self):
        """
        Prepares for a new episode.
        We sample a new random batch of data (e.g., 100 people)
        to simulate stochasticity. 
        """
        obs, _ = super().reset()
        self.sampled_df = self.full_df.sample(n=100).copy()
        self.selected_pipeline_cols = []
        return obs, {}

    def step(self, action):
        """
        Executes one step of the simulation.
        Runs the RDDL physics (state transitions).
        Maps the RDDL action to a DataFrame column.
        Calculates the Penalty (WRS) if the pipeline is finished.
        """
        # Run RDDL Physics
        obs, reward, terminated, truncated, info = super().step(action)
        
        done = terminated or truncated
        is_pipeline_done = obs.get("pipeline_done", 0) >= 1
        
        # Identify the Action
        # RDDL actions come as dicts: {'select_model___m1': 1}
        selected_model_rddl = None
        for k, v in action.items():
            if v == 1 and "select_model" in k:
                selected_model_rddl = k.split("___")[-1] # Extract 'm1'
                break
        
        # Map to Data
        # We need to find the correct column in the CSV (e.g., 's1_m1').
        if selected_model_rddl:
            #The environment state has already updated to the NEXT stage.
            # We need to know which stage we acted on.
            # The number of columns we've collected so far tells us the index.
            current_step_idx = len(self.selected_pipeline_cols)
            
            if current_step_idx < len(self.stage_model_map):
                # Get stage name by index (e.g. 0 -> 's1', 1 -> 's2')
                # We sort keys to ensure s1, s2, s3 order is respected.
                sorted_stages = sorted(list(self.stage_model_map.keys()), key=lambda x: int(x[1:]))
                current_stage = sorted_stages[current_step_idx]
                
                # Construct Column Name: "s1_m1"
                col_name = f"{current_stage}_{selected_model_rddl}"
                
                if col_name in self.full_df.columns:
                    self.selected_pipeline_cols.append(col_name)

        # Calculate Reward
        # Only apply the penalty if the pipeline is complete and we have data.
        if is_pipeline_done and len(self.selected_pipeline_cols) > 0:
            final_col = self.selected_pipeline_cols[-1]
            
            total_wrs = 0.0
            try:
                # Sum WRS for all protected attributes (Race, Gender, etc.)
                for z in ['Z1', 'Z2', 'Z3']:
                    total_wrs += calc_wrs(self.sampled_df, z, final_col)
            except Exception as e:
                # If metric fails (e.g. empty group), ignore penalty to prevent crash
                pass
            
            # Apply Penalty: Higher WRS = Lower Reward.
            # We scale by 10.0 to make the signal strong enough for the agent to care.
            reward -= (total_wrs * 10.0)
            
            done = True

        return obs, reward, done, truncated, info