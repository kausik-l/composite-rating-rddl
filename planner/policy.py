import numpy as np
import random

class PipelineQPlanner:
    """
    The Q-Learning Agent.
    
    This agent learns from trial and error. 
    It has a memory (Q-Table) that tracks which models perform best at each stage.
    It balances 'Exploration' with 'Exploitation' .
    """
    def __init__(self, action_space, stage_map, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Args:
            stage_map: Dictionary mapping Stage -> List of Valid Models.
            alpha: Learning Rate (0.1). How much we trust new information vs old memory.
            gamma: Discount Factor (0.99). How much we care about future rewards.
            epsilon: Exploration Rate (0.1). Probability of taking a random action.
        """
        # Stores values like q_table['s1']['m11'] = -12.5
        self.q_table = {} 
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.action_space = action_space
        self.stage_map = stage_map 

    def get_current_stage(self, state):
        """
        Parses the RDDL state dictionary to find where we are (e.g., 's1').
        State dict looks like: {'current_stage___s1': True, ...}
        """
        for key, val in state.items():
            if "current_stage" in key and val == True:
                # Extract the stage name from the variable key
                return key.split("___")[-1] 
        return None

    def sample_action(self, state):
        """
        Decides the next move using Epsilon-Greedy Strategy.
        """
        # Where are we?
        current_stage = self.get_current_stage(state)
        
        # If stage is unknown or invalid, do nothing
        if not current_stage or current_stage not in self.stage_map:
            return {}

        valid_models = self.stage_map[current_stage]
        
        # Initialize Memory if this is a new stage
        if current_stage not in self.q_table:
            self.q_table[current_stage] = {m: 0.0 for m in valid_models}

        # Make a Choice
        # Exploration: Roll a die. If < epsilon, pick purely randomly.
        if random.uniform(0, 1) < self.epsilon:
            chosen_model = random.choice(valid_models)
        
        # Exploitation: Otherwise, pick the model with the highest Q-Value.
        else:
            qs = self.q_table[current_stage]
            # Find key with max value. If tie, this picks the first one.
            chosen_model = max(qs, key=qs.get)

        # Return properly formatted action
        return {f"select_model___{chosen_model}": 1}

    def update(self, state, action, reward, next_state):
        """
        Bellman Equation.
        Updates the Q-Value of the action we just took based on the reward we got.
        """
        # Identify Context
        curr_stage = self.get_current_stage(state)
        next_stage_id = self.get_current_stage(next_state)
        
        # Parse the action we took to find the model name 
        used_model = None
        for k, v in action.items():
            if v == 1 and "select_model" in k:
                used_model = k.split("___")[-1]
                break
        
        if curr_stage and used_model:
            # Make sure entries exist (in case update runs before sample)
            if curr_stage not in self.q_table:
                 self.q_table[curr_stage] = {m: 0.0 for m in self.stage_map[curr_stage]}
            if used_model not in self.q_table[curr_stage]:
                 self.q_table[curr_stage][used_model] = 0.0

            # Get Old Value
            old_q = self.q_table[curr_stage][used_model]
            
            # Estimate Future Value (Max Q of the next state)
            next_max_q = 0.0
            if next_stage_id and next_stage_id in self.q_table:
                # What is the best I can do from the next stage?
                next_max_q = max(self.q_table[next_stage_id].values())
            
            # Update Rule
            # New = Old + Alpha * (Target - Old)
            # Target = Immediate Reward + Discounted Future Value
            target = reward + self.gamma * next_max_q
            new_q = old_q + self.alpha * (target - old_q)
            
            self.q_table[curr_stage][used_model] = new_q