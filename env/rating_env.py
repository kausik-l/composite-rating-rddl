from pyRDDLGym.core.env import RDDLEnv
import pandas as pd
from env.metric_utils import calc_wrs

class RatingEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_files, subset_size=100, metric_weights=None):
        super().__init__(domain_file, instance_file)

        # Load datasets for each plan (english, french, roundtrip)
        self.datasets = {k: pd.read_csv(v) for k, v in data_files.items()}

        # Subset size for evaluation
        self.subset_size = subset_size

        # Metric weights (default: only WRS matters)
        self.metric_weights = metric_weights or {"wrs": 1.0}

    def step(self, action_dict):
        next_state, _, done, truncated, info = super().step(action_dict)

        # Which action(s) were chosen?
        chosen = [a for a, v in action_dict.items() if v == 1]
        if not chosen:
            return next_state, -999, True, truncated, info
        chosen_action = chosen[0]

        # --- Case 1: Translate actions ---
        if "do_translate" in chosen_action:
            # No sentiment output → no WRS
            reward = 0.0
            info["wrs"] = 0.0
            info["pie"] = 0.0
            info["ate"] = 0.0
            return next_state, reward, done, truncated, info

        # --- Case 2: Sentiment actions ---
        if "do_sentiment_english" in chosen_action:
            df = self.datasets["english"]
        elif "do_sentiment_french" in chosen_action:
            df = self.datasets["french"]
        else:
            # fallback
            df = self.datasets["english"]

        # Sample subset
        subset = df.sample(n=min(self.subset_size, len(df)))

        # Compute metrics
        wrs = calc_wrs(subset, protected_col="gender", output_col="sentiment_outcome")
        pie = 0.0
        ate = 0.0

        # count how many actions were chosen in this step
        n_actions = sum(v for v in action_dict.values())
        action_cost = n_actions * self.metric_weights.get("action_cost", 0.0)

        # total reward = benefit - cost
        reward = -(self.metric_weights["wrs"] * wrs +
                self.metric_weights.get("pie", 0.0) * pie +
                self.metric_weights.get("ate", 0.0) * ate) - action_cost


        info["wrs"] = wrs
        info["pie"] = pie
        info["ate"] = ate

        return next_state, reward, done, truncated, info

