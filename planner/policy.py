# planner/policy.py
import random

class SimplePolicy:
    """
    Extremely small policy for quick testing.
    - Scans env.state for keys starting with 'model_used'
    - Picks the first (stage,input,model) whose corresponding processed(stage,input) is False
    - Returns action dict with the same flattening, by replacing 'model_used' -> 'choose_model'
    """

    def __init__(self, env, seed=None):
        self.env = env
        if seed is not None:
            random.seed(seed)

    def get_action(self, state):
        state = state or {}
        # find all model_used keys (flattened form). Keep order deterministic.
        keys = sorted(k for k in state.keys() if k.startswith("model_used"))
        for mu_key in keys:
            # example mu_key: model_used___s1__dataset__m11
            parts = mu_key.split("__")
            # Accept both 4-part and 5+ part splits (env flattening may produce extra leading underscores)
            if len(parts) < 4:
                continue
            # last three parts are expected to be s, i, m (may contain leading underscores)
            s = parts[-3].strip("_")
            i = parts[-2].strip("_")
            m = parts[-1].strip("_")
            if not s or not i or not m:
                continue
            # check processed using several common variants (flattened or paren)
            proc_flat1 = f"processed__{s}__{i}"
            proc_flat2 = f"processed___{s}__{i}"
            proc_paren = f"processed({s},{i})"
            if state.get(proc_flat1, False) or state.get(proc_flat2, False) or state.get(proc_paren, False):
                # already processed, skip
                continue
            # build action key using same prefix/flattening as the observed model_used key
            action_key = mu_key.replace("model_used", "choose_model", 1)
            return {action_key: True}
        # nothing to do
        return {}
