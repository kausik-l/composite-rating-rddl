from pyRDDLGym.core.env import RDDLEnv
import pandas as pd
from env.metric_utils import calc_wrs  # (later you'll add calc_pie, calc_ate, etc.)

# --------------------------------------------------------------------
# RatingEnv is a custom environment that extends pyRDDLGym's RDDLEnv.
# It overrides the reward to use ARC metrics (WRS, PIE, ATE)
# --------------------------------------------------------------------
class RatingEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_files, subset_size=100, metric_weights=None):
        # Initialize the RDDL domain and instance.
        super().__init__(domain_file, instance_file)

        # Load all datasets (english, french, roundtrip).
        # Each dataset represents outcomes for one pipeline plan.
        self.datasets = {k: pd.read_csv(v) for k, v in data_files.items()}

        # Instead of scoring the full dataset every time, 
        # we sample random subsets.
        self.subset_size = subset_size

        # Define which metrics contribute to reward and their weights.
        # By default, we are only using WRS for now (weight=1.0).
        self.metric_weights = metric_weights or {"wrs": 1.0}

    def step(self, action):
        # First, let RDDL update the state
        #   (e.g., set has_sentiment=True if do_sentiment_english is chosen).
        # We throw away the RDDL reward (always 0.0 in our domain).
        next_state, rew, done, truncated, info = super().step(action)

        # ----------------------------------------------------------------
        # Link action to the dataset:
        #   If the agent chooses English sentiment, evaluate on english.csv.
        #   If French sentiment, evaluate on french.csv.
        #   Otherwise (e.g., roundtrip plan), evaluate on roundtrip.csv.
        #
        # This mapping is crude here (based on action name),
        # but captures the idea: different pipelines → different data outputs.
        # ----------------------------------------------------------------
        if "do_sentiment_english" in action:
            df = self.datasets["english"]
        elif "do_sentiment_french" in action:
            df = self.datasets["french"]
        else:
            df = self.datasets["roundtrip"]

        # ----------------------------------------------------------------
        # Simulate by sampling a subset of the dataset instead of always 
        # using the full data.
        # This introduces some variation episode-to-episode.
        # ----------------------------------------------------------------
        subset = df.sample(n=min(self.subset_size, len(df)))

        # ----------------------------------------------------------------
        # Compute ARC metrics on the subset.
        # Currently:
        #   - WRS (Weighted Rejection Score) is implemented.
        #   - PIE and ATE are just placeholders for now.
        # ----------------------------------------------------------------
        wrs = calc_wrs(subset, protected_col="gender", output_col="sentiment_outcome")
        pie = 0.0
        ate = 0.0

        # ----------------------------------------------------------------
        # Combine metrics into a single reward.
        # reward = - WRS (less bias means higher reward).
        # ----------------------------------------------------------------
        reward = -(self.metric_weights["wrs"] * wrs +
                   self.metric_weights.get("pie", 0.0) * pie +
                   self.metric_weights.get("ate", 0.0) * ate)

        # Attach metrics to info dict for logging and plotting later (see utils folder).
        info["wrs"] = wrs
        info["pie"] = pie
        info["ate"] = ate

        return next_state, reward, done, truncated, info
