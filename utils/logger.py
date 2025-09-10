import json
import numpy as np

class Logger:
    def __init__(self, log_dir="data/output", log_file="simulation_log.json"):
        self.log_path = f"{log_dir}/{log_file}"
        self.episodes = []

    def log_episode(self, episode_idx, steps, total_reward, final_state):
        self.episodes.append({
            "episode": episode_idx,
            "steps": steps,
            "total_reward": float(total_reward),
            "final_state": self._convert(final_state)
        })

    def _convert(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        else:
            return obj

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.episodes, f, indent=2)
        print(f"Logs saved to {self.log_path}")
