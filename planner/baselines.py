import random

class RandomPipelinePlanner:
    """
    Baseline 1: Random Agent.
    Represents the lower bound of performance (Noise).
    It ignores costs and fairness, switching families randomly.
    """
    def __init__(self, stage_map):
        self.stage_map = stage_map

    def sample_action(self, state):
        current_stage = None
        for key, val in state.items():
            if "current_stage" in key and val == True:
                current_stage = key.split("___")[-1]
                break
        
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage]
            choice = random.choice(options)
            return {f"select_model___{choice}": 1}
        return {}

    def update(self, state, action, reward, next_state):
        pass # No learning


class FixedPipelinePlanner:
    """
    Baseline 2: Fixed Policy Agent.
    Represents a rigid, legacy system that never adapts.
    It always picks the Nth model (e.g., Always Model 1).
    It effectively stays in one 'Family' forever.
    """
    def __init__(self, stage_map, selection_index=0, name="Fixed"):
        self.stage_map = stage_map
        self.selection_index = selection_index
        self.name = name

    def sample_action(self, state):
        current_stage = None
        for key, val in state.items():
            if "current_stage" in key and val == True:
                current_stage = key.split("___")[-1]
                break
        
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage]
            # Safety fallback
            idx = min(self.selection_index, len(options) - 1)
            choice = options[idx]
            return {f"select_model___{choice}": 1}
        return {}

    def update(self, state, action, reward, next_state):
        pass


class BestFirstSearchPlanner:
    """
    Baseline 3: Greedy Heuristic Agent.
    Represents a 'smart' but short-sighted system.
    It looks at the previous family used and blindly picks a model 
    from the same family to avoid the immediate switching cost.
    It ignores global fairness.
    """
    def __init__(self, stage_map, name="BestFirst"):
        self.stage_map = stage_map
        self.name = name

    def _get_active_features(self, state):
        stage = None
        last_fam = None
        for key, val in state.items():
            if val == True:
                if "current_stage" in key:
                    stage = key.split("___")[-1]
                elif "last_used_family" in key:
                    last_fam = key.split("___")[-1]
        return stage, last_fam

    def sample_action(self, state):
        current_stage, last_family = self._get_active_features(state)
        
        if not current_stage or current_stage not in self.stage_map:
            return {}

        options = self.stage_map[current_stage]
        
        # Greedy Heuristic: Match the family
        if not last_family or "none" in last_family:
            choice = options[0]
        elif "fam_1" in last_family:
            # Pick odd index (0, 2...)
            candidates = [opt for i, opt in enumerate(options) if i % 2 == 0]
            choice = candidates[0] if candidates else options[0]
        elif "fam_2" in last_family:
            # Pick even index (1, 3...)
            candidates = [opt for i, opt in enumerate(options) if i % 2 != 0]
            choice = candidates[0] if candidates else options[0]
        else:
            choice = random.choice(options)

        return {f"select_model___{choice}": 1}

    def update(self, state, action, reward, next_state):
        pass