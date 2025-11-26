import numpy as np
import random

class ContextAwareQPlanner:
    """
    Intelligent Agent: Context-Aware Q-Learning.
    
    Unlike standard planners, this agent incorporates 'Context' (History) into its state.
    It tracks not just the current stage, but also the 'Family' of the model used previously.
    
    This allows it to learn 'Lanes': "If I am in the CPU lane, I should stay in the CPU lane."
    It balances this inertia against the need to find the Fair Family.
    """
    def __init__(self, action_space, stage_map, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = {} 
        self.alpha = alpha      # Learning Rate
        self.gamma = gamma      # Discount Factor
        self.epsilon = epsilon  # Exploration Rate
        
        self.action_space = action_space
        self.stage_map = stage_map 

    def _parse_rddl_state(self, state):
        """Extracts (CurrentStage, LastFamily) from the raw state dict."""
        curr_stage = None
        last_fam = "unknown"
        
        for key, val in state.items():
            if val == True or val == 1:
                if "current_stage" in key:
                    curr_stage = key.split("___")[-1]
                elif "last_used_family" in key:
                    last_fam = key.split("___")[-1]
        
        return curr_stage, last_fam

    def get_state_key(self, state):
        """Unique String Key for the Q-Table: 's1__fam_1'"""
        s, f = self._parse_rddl_state(state)
        if not s: return "DONE"
        return f"{s}__{f}"

    def sample_action(self, state):
        """Epsilon-Greedy Action Selection."""
        state_key = self.get_state_key(state)
        stage, _ = self._parse_rddl_state(state)
        
        if state_key == "DONE" or stage not in self.stage_map:
            return {}

        valid_models = self.stage_map[stage]
        
        # Init Q-Table if new state encountered
        if state_key not in self.q_table:
            self.q_table[state_key] = {m: 0.0 for m in valid_models}

        # Exploration (Random)
        if random.uniform(0, 1) < self.epsilon:
            chosen = random.choice(valid_models)
        # Exploitation (Best Known)
        else:
            qs = self.q_table[state_key]
            chosen = max(qs, key=qs.get)

        return {f"select_model___{chosen}": 1}

    def update(self, state, action, reward, next_state):
        """Standard Bellman Update."""
        curr_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        
        # Parse action string to get model name
        used_model = None
        for k, v in action.items():
            if v == 1 and "select_model" in k:
                used_model = k.split("___")[-1]
                break
        
        if curr_key != "DONE" and used_model:
            if curr_key not in self.q_table:
                stage, _ = self._parse_rddl_state(state)
                self.q_table[curr_key] = {m: 0.0 for m in self.stage_map[stage]}

            old_q = self.q_table[curr_key][used_model]
            
            # Calculate Max Q for next state
            next_max_q = 0.0
            if next_key != "DONE":
                if next_key not in self.q_table:
                    n_stage, _ = self._parse_rddl_state(next_state)
                    if n_stage:
                        self.q_table[next_key] = {m: 0.0 for m in self.stage_map[n_stage]}
                
                if next_key in self.q_table:
                    next_max_q = max(self.q_table[next_key].values())

            # Update
            new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
            self.q_table[curr_key][used_model] = new_q