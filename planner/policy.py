import random

# --------------------------------------------------------------------
# Base class for all policies. A "policy" here means:
#   Given the current state of the environment, which PLAN (pipeline)
#   should we choose to execute? 
# Plans are high-level for now: "english", "french", "roundtrip".
# --------------------------------------------------------------------
class BasePolicy:
    def __init__(self, env, plan_to_actions):
        """
        env: the RatingEnv (wraps pyRDDLGym env + WRS).
        plan_to_actions: dictionary that maps each plan name
                         to the list of RDDL actions that 
                         represent that plan.
            Example:
              {
                "english": ["do_sentiment_english___t1"],
                "french": ["do_sentiment_french___t1"],
                "roundtrip": [
                   "do_translate___t1__English__French",
                   "do_translate___t1__French__English",
                   "do_sentiment_english___t1"
                 ]
              }
        """
        self.env = env
        self.plan_to_actions = plan_to_actions
        # A list of all possible plans.
        self.possible_plans = list(plan_to_actions.keys())

    def select_plan(self, state):
        """Each specific policy will implement its own logic for 
        selecting one plan out of self.possible_plans."""
        raise NotImplementedError

    def build_action_dict(self, plan):
        """
        Convert a chosen high-level plan into a dictionary of actions.
        pyRDDLGym expects env.step() to receive a dictionary where:
          - keys = all possible actions
          - values = 0 (inactive) or 1 (active this step)
        So here:
          1. Initialize everything to 0.
          2. Turn on (=1) only those actions that correspond to the chosen plan.
        """
        # Start with all actions "off".
        action_dict = {name: 0 for name in self.env.action_space.spaces.keys()}
        # Switch on the ones belonging to this plan.
        for act in self.plan_to_actions[plan]:
            action_dict[act] = 1
        return action_dict


# --------------------------------------------------------------------
# Policy that just picks randomly between available plans.
# --------------------------------------------------------------------
class RandomPolicy(BasePolicy):
    def select_plan(self, state):
        return random.choice(self.possible_plans)


# --------------------------------------------------------------------
# Policy that always returns the same plan, no matter the state.
# --------------------------------------------------------------------
class FixedPolicy(BasePolicy):
    def __init__(self, env, plan_to_actions, fixed_plan):
        super().__init__(env, plan_to_actions)
        self.fixed_plan = fixed_plan   # e.g., "english"

    def select_plan(self, state):
        return self.fixed_plan


# --------------------------------------------------------------------
# Greedy policy:
#   - Looks at all possible plans.
#   - For each plan, it calls env.simulate(action_dict) to estimate
#     what the immediate reward would be if we chose it.
#   - Picks the plan with the highest reward.
#
# Note: This requires that the environment implements a simulate()
# method, which runs a mock step and then restores the state.
# This way we do not permanently advance the environment while evaluating.
# --------------------------------------------------------------------
class GreedyPolicy(BasePolicy):
    def select_plan(self, state):
        best_plan, best_reward = None, float("-inf")
        for plan in self.possible_plans:
            # Build the action dictionary for this plan.
            action_dict = self.build_action_dict(plan)
            # Run a simulated step to see what reward it would give.
            next_state, reward, done, truncated, info = self.env.simulate(action_dict)
            # Keep track of the best one.
            if reward > best_reward:
                best_plan, best_reward = plan, reward
        return best_plan
