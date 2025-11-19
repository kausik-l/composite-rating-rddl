# env/rating_env.py
"""
Assumptions / design:
- The user provides one CSV (data_files={"default": ...}).
  That CSV must include model output columns named like "s1_m11", "s2_m21", ...
  and protected columns like "Z1", "Z2", "Z3" (configure via protected_cols).
- We then process the action dict returned by the planner, find the first chosen
  (stage,input,model) for which the environment recorded model_used(...)=True,
  and compute WRS on the corresponding CSV column "<stage>_<model>".
- If the expected column is missing, then return a large negative penalty.
"""

from pyRDDLGym.core.env import RDDLEnv
import pandas as pd
import numpy as np
from env.metric_utils import calc_wrs


class RatingEnv(RDDLEnv):
    # Penalty when the expected column (e.g. "s2_m11") is not present in the CSV.
    MISSING_OUTPUT_PENALTY = -10.0

    def __init__(self, domain_file, instance_file, data_files, subset_size=100, protected_cols=("Z1", "Z2", "Z3"), metric_weights=None,):

        super().__init__(domain_file, instance_file)

        # Expect exactly one dataset under 'default' (I just added this to make it easy to allow multiple datasets at once in the future).  
        default = data_files["default"]
        self.df = pd.read_csv(default) if isinstance(default, str) else default.copy()

        # Config.
        self.subset_size = int(subset_size)
        self.protected_cols = list(protected_cols)
        # TODO: Weight of each metric can be adjusted here, if we later want a cumulative reward. 
        self.metric_weights = metric_weights or {"wrs": 1.0}

        # A lot of Print statements are sprinkled throughout the code to help us debug.
        print(f"[INIT] Loaded data shape: {self.df.shape}")
        print(f"[INIT] Columns: {list(self.df.columns)}")


    # We need to extract the stage, input, model from the default PyRDDLGym action format. 
    def _parse_choose_model(self, key):
        parts = key.split("__")
        return parts[-3].strip("_"), parts[-2].strip("_"), parts[-1].strip("_")


    def _model_used(self, state, s, i, m):
        """
        We only check the two flattened forms that pyRDDLGym commonly creates.
        """
        key1 = f"model_used___{s}__{i}__{m}"
        return bool(state.get(key1, False))


    def step(self, action_dict):
        """
        Compute a reward derived from WRS.

        Steps:
        - Unpack the 5-tuple returned by pyRDDLGym's step (next_state, reward, done, truncated, info).
        - Get chosen actions from action_dict, parse them, and find the first chosen model
          that the RDDL state recorded as used.
        - For that chosen model, get a column "<stage>_<model>" in self.df (our test dataset); sample rows,
          set "__MODEL_OUTPUT__", compute WRS for each protected_col using calc_wrs,
          average the WRS values into mean_wrs.
        - Reward = - mean_wrs * weight - action_cost.
        - If the expected column is missing, return MISSING_OUTPUT_PENALTY - action_cost immediately.
        """
        # PyRDDLGym returns a 5-tuple.
        next_state, _, done, truncated, info = super().step(action_dict)
        info = dict(info or {})

        # parse chosen actions into (s,i,m)
        chosen = []
        for k, v in (action_dict or {}).items():
            if not bool(v):
                continue
            parsed = self._parse_choose_model(k)
            if parsed:
                chosen.append(parsed)

        print(f"[STEP] Chosen actions (parsed): {chosen}")

        # small penalty for no-op
        if not chosen:
            info["_debug_action_cost"] = 0.0
            return next_state, -0.1, done, truncated, info

        action_cost = float(len(chosen)) * 1.0

        # find the first chosen model that the RDDL state recorded as used
        chosen_model = None
        for (s, i, m) in chosen:
            # we expect the environment to write model_used__s__i__m or model_used___s__i__m
            if self._model_used(next_state, s, i, m):
                chosen_model = (s, i, m)
                break

        # if nothing was actually used, charge action cost and return
        if chosen_model is None:
            print("[STEP] No model_used flag found for chosen actions -> action cost only")
            info["_debug_action_cost"] = action_cost
            info["wrs_per_protected"] = {p: None for p in self.protected_cols}
            info["wrs_mean"] = None
            return next_state, -action_cost, done, truncated, info

        s, i, m = chosen_model
        print(f"[STEP] Evaluating model_used: stage={s}, input={i}, model={m}")

        # expected CSV column, e.g., "s1_m11"
        col = f"{s}_{m}"
        if col not in self.df.columns:
            # immediate penalty when missing the expected column — loud and clear
            print(f"[ERROR] Missing expected column in CSV: '{col}'")
            info["_missing_output_column_for"] = f"{s},{i},{m}"
            info["_available_columns"] = list(self.df.columns)
            info["_debug_action_cost"] = action_cost
            info["wrs_per_protected"] = {p: None for p in self.protected_cols}
            info["wrs_mean"] = None
            return next_state, self.MISSING_OUTPUT_PENALTY - action_cost, done, truncated, info

        # sample rows (simple sampling)
        n = min(self.subset_size, len(self.df))
        sample_df = self.df.sample(n=n, replace=False) if n < len(self.df) else self.df.copy()
        sample_df = sample_df.copy()
        sample_df["__MODEL_OUTPUT__"] = sample_df[col]
        print(f"[STEP] Sampled {len(sample_df)} rows from column '{col}' for WRS")

        # compute WRS for each protected column and average
        wrs_values = []
        wrs_by_col = {}
        for p in self.protected_cols:
            val = float(calc_wrs(sample_df, protected_col=p, output_col="__MODEL_OUTPUT__"))
            wrs_by_col[p] = val
            wrs_values.append(val)
            print(f"[WRS] {p} -> {val}")

        mean_wrs = float(np.mean(wrs_values)) if wrs_values else 0.0

        # final reward: negative mean WRS minus simple action cost
        reward = - self.metric_weights.get("wrs", 1.0) * mean_wrs - action_cost

        print(f"[RESULT] mean_wrs={mean_wrs}, action_cost={action_cost}, reward={reward}")

        # fill info dict for downstream logging
        info["wrs_per_protected"] = wrs_by_col
        info["wrs_mean"] = mean_wrs
        info["_debug_action_cost"] = action_cost
        info["_debug_total_sampled"] = len(sample_df)

        return next_state, reward, done, truncated, info
