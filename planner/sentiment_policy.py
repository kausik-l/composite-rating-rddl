import numpy as np
import random
import pickle
import os

class ContextAwareQPlanner:
    """
    Intelligent Agent: Context-Aware Q-Learning.
    Tracks state context (Stage + LastFamily) to learn optimal paths.
    """
    def __init__(self, action_space, stage_map, alpha=0.1, gamma=0.99, epsilon=0.1, action_name="select_model"):
        self.q_table = {} 
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon  
        
        self.action_space = action_space
        self.stage_map = stage_map
        self.action_name = action_name # Configurable action fluent name
        
        self.training_history = [] 

    def _parse_rddl_state(self, state):
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
        s, f = self._parse_rddl_state(state)
        if not s: return "DONE"
        return f"{s}__{f}"

    def sample_action(self, state):
        state_key = self.get_state_key(state)
        stage, _ = self._parse_rddl_state(state)
        
        if state_key == "DONE" or stage not in self.stage_map:
            return {}

        valid_models = self.stage_map[stage]
        if state_key not in self.q_table:
            self.q_table[state_key] = {m: 0.0 for m in valid_models}

        if random.uniform(0, 1) < self.epsilon:
            chosen = random.choice(valid_models)
        else:
            qs = self.q_table[state_key]
            chosen = max(qs, key=qs.get)

        # Use the configured action name
        return {f"{self.action_name}___{chosen}": 1}

    def update(self, state, action, reward, next_state):
        curr_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        
        used_model = None
        for k, v in action.items():
            if v == 1 and self.action_name in k:
                used_model = k.split("___")[-1]
                break
        
        if curr_key != "DONE" and used_model:
            if curr_key not in self.q_table:
                stage, _ = self._parse_rddl_state(state)
                # Safety check: stage map access
                if stage in self.stage_map:
                    self.q_table[curr_key] = {m: 0.0 for m in self.stage_map[stage]}
                else:
                    return # Should not happen if logic flows correctly

            if used_model in self.q_table[curr_key]:
                old_q = self.q_table[curr_key][used_model]
                
                next_max_q = 0.0
                if next_key != "DONE":
                    if next_key not in self.q_table:
                        n_stage, _ = self._parse_rddl_state(next_state)
                        if n_stage and n_stage in self.stage_map:
                            self.q_table[next_key] = {m: 0.0 for m in self.stage_map[n_stage]}
                    
                    if next_key in self.q_table:
                        next_max_q = max(self.q_table[next_key].values())

                new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
                self.q_table[curr_key][used_model] = new_q

    def save_agent(self, filepath="q_agent.pkl"):
        data = {
            "q_table": self.q_table,
            "hyperparams": {"alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon},
            "history": self.training_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")

    def load_agent(self, filepath="q_agent.pkl"):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data["q_table"]
            self.training_history = data.get("history", [])
            print(f"Agent loaded from {filepath} (History: {len(self.training_history)} eps)")
            return True
        else:
            return False