import random

# --------------------------------------------------------------------
# Base class for all policies. A "policy" here means:
#   Given the current state of the environment, which PLAN (pipeline)
#   should we choose to execute? 
# Plans are high-level: "english", "french", "roundtrip".
# Each plan is represented as a *list of steps*,
# where each step is a list of RDDL actions.
#
# Example:
#   {
#     "english": [["do_sentiment_english___t1"]],
#     "french": [["do_sentiment_french___t1"]],
#     "roundtrip": [
#        ["do_translate___t1__English__French"],
#        ["do_translate___t1__French__English"],
#        ["do_sentiment_english___t1"]
#     ]
#   }
# --------------------------------------------------------------------
class BasePolicy:
    def __init__(self, env, possible_plans):
        self.env = env
        self.possible_plans = possible_plans

    def select_plan(self, state):
        raise NotImplementedError




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
#   - For each plan, it simulates the full sequence of steps.
#   - Picks the plan with the highest cumulative reward.
#
# This requires the environment to implement simulate(),
# which runs a step without committing state permanently.
# --------------------------------------------------------------------
class GreedyPolicy(BasePolicy):
    def select_plan(self, state):
        best_plan, best_reward = None, float("-inf")

        for plan in self.possible_plans:
            total_reward = 0
            # Save state once before simulating whole plan
            saved_state = self.env.sampler.copy_state(self.env.sampler.state)

            # Simulate each step in the plan
            for step_actions in self.plan_to_actions[plan]:
                action_dict = self.build_action_dict(step_actions)
                _, reward, done, truncated, _ = self.env.simulate(action_dict)
                total_reward += reward
                if done or truncated:
                    break

            # Restore state
            self.env.sampler.state = saved_state

            if total_reward > best_reward:
                best_plan, best_reward = plan, total_reward

        return best_plan
