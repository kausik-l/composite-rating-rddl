from pyRDDLGym.core.env import RDDLEnv
import pandas as pd
from env.metric_utils import calc_wrs  # you’ll later add calc_pie, calc_ate, etc.

class RatingEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_files, subset_size=100, metric_weights=None):
        super().__init__(domain_file, instance_file)
        self.datasets = {k: pd.read_csv(v) for k, v in data_files.items()}
        self.subset_size = subset_size
        # default: only WRS matters
        self.metric_weights = metric_weights or {"wrs": 1.0}

    def step(self, action):
        # Delegate to parent for state transition
        next_state, _, done, truncated, info = super().step(action)

        # --- Map action to dataset (simple for now, improve later) ---
        if "do_sentiment_english" in action:
            df = self.datasets["english"]
        elif "do_sentiment_french" in action:
            df = self.datasets["french"]
        else:
            df = self.datasets["roundtrip"]

        # --- Sample subset ---
        subset = df.sample(n=min(self.subset_size, len(df)))

        # --- Compute metrics ---
        wrs = calc_wrs(subset, protected_col="gender", output_col="sentiment_outcome")
        pie = 0.0  # placeholder for PIE
        ate = 0.0  # placeholder for ATE

        # --- Weighted reward ---
        reward = -(self.metric_weights["wrs"] * wrs +
                   self.metric_weights.get("pie", 0.0) * pie +
                   self.metric_weights.get("ate", 0.0) * ate)

        info["wrs"] = wrs
        info["pie"] = pie
        info["ate"] = ate
        return next_state, reward, done, truncated, info
