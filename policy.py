from pyRDDLGym.core.policy import BaseAgent

class DoSomething(BaseAgent):
    def __init__(self, action_space, num_actions):
        self.action_space = action_space
        self.num_actions = num_actions

    def sample_action(self, obs):
        # Return one true-valued action per step for simplicity
	# If the action name starts with 'Do_', it includes it in the dictionary.
        actions = {}
        count = 0
        for key in self.action_space:
            if 'Do_' in key[0] and count < self.num_actions:
                actions[key] = True
                count += 1
        return actions

def build_policy(env):
    return DoSomething(env.action_space, env.max_allowed_actions)
