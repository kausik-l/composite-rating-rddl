import random

class RandomPipelinePlanner:
    """
    Baseline 1: The Random Agent.    
    It simply ensures that it picks a valid move / action at each stage.
    """
    def __init__(self, stage_map):
        """
        Args:
            stage_map: A dictionary mapping Stage -> List of Valid Models.
                       Example: {'s1': ['m1', 'm2'], 's2': ['m1', 'm2']}
        """
        self.stage_map = stage_map

    def sample_action(self, state):
        """
        Decides which action to take based on the current state.
        """
        # Identify Where We Are
        # The state dictionary looks like: {'current_stage___s1': True, ...}
        # We need to parse this to find the string "s1".
        current_stage = None
        for key, val in state.items():
            # We look for the active stage flag (value is True or 1)
            if "current_stage" in key and val == True:
                # Extract "s1" from "current_stage___s1"
                current_stage = key.split("___")[-1]
                break
        
        # Pick a Random Valid Model
        # We use the stage_map to ensure we don't pick an illegal action.
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage] # e.g., ['m1', 'm2']
            choice = random.choice(options)         # Randomly pick one
            
            # Return in PyRDDLGym action format
            return {f"select_model___{choice}": 1}
        
        # If we can't find the stage (e.g., pipeline is done), do nothing.
        return {}

    def update(self, state, action, reward, next_state):
        """
        Random agent ignores rewards and never learns, so this function is empty.
        """
        pass


class FixedPipelinePlanner:
    """
    Baseline 2: The Fixed Policy Agent.
    
    This agent follows a strict, hardcoded rule.
    It doesn't adapt to data.
    """
    def __init__(self, stage_map, selection_index=0, name="Fixed"):
        """
        Args:
            selection_index: Which model index to always pick.
                             0 = Always pick the 1st model (m1)
                             1 = Always pick the 2nd model (m2)
        """
        self.stage_map = stage_map
        self.selection_index = selection_index
        self.name = name

    def sample_action(self, state):
        # Identify Current Stage (Same logic as above)
        current_stage = None
        for key, val in state.items():
            if "current_stage" in key and val == True:
                current_stage = key.split("___")[-1]
                break
        
        # Pick the Pre-Determined Model
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage]
            
            # If we want the 5th model but only 3 exist, pick the last one.
            # This prevents "IndexError".
            idx = min(self.selection_index, len(options) - 1)
            
            choice = options[idx]
            return {f"select_model___{choice}": 1}
        
        return {}

    def update(self, state, action, reward, next_state):
        # Fixed Agent ignores feedback and never gets updated.
        pass