import random

class BasePolicy:
    """Abstract base class for all policies (work at the PLAN level)."""

    def __init__(self, env, plan_to_actions):
        """
        env: the CustomRatingEnv (wraps pyRDDLGym env).
        plan_to_actions: dict mapping plan names -> list of grounded actions
                         e.g., {"english": ["do_sentiment_english___t1"], ...}
        """
        self.env = env
        self.plan_to_actions = plan_to_actions
        self.possible_plans = list(plan_to_actions.keys())

    def select_plan(self, state):
        """Choose a plan (english/french/roundtrip). Must be overridden."""
        raise NotImplementedError

    def build_action_dict(self, plan):
        """Convert a chosen plan into a valid action dictionary for env.step()."""
        action_dict = {name: 0 for name in self.env.action_space.spaces.keys()}
        for act in self.plan_to_actions[plan]:
            action_dict[act] = 1
        return action_dict


class RandomPolicy(BasePolicy):
    """Selects a random plan."""
    def select_plan(self, state):
        return random.choice(self.possible_plans)


class FixedPolicy(BasePolicy):
    """Always selects the same plan (useful for debugging)."""
    def __init__(self, env, plan_to_actions, fixed_plan):
        super().__init__(env, plan_to_actions)
        self.fixed_plan = fixed_plan

    def select_plan(self, state):
        return self.fixed_plan


class GreedyPolicy(BasePolicy):
    """
    Selects the plan that maximizes immediate reward (one-step lookahead).
    NOTE: requires env.simulate(action_dict) to be implemented in CustomRatingEnv.
    """
    def select_plan(self, state):
        best_plan, best_reward = None, float("-inf")
        for plan in self.possible_plans:
            action_dict = self.build_action_dict(plan)
            next_state, reward, done, truncated, info = self.env.simulate(action_dict)
            if reward > best_reward:
                best_plan, best_reward = plan, reward
        return best_plan
