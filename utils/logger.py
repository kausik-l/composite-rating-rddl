import json
import os
import numpy as np

class Logger:
    def __init__(self, log_dir="data/output", log_file="simulation_log.json"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.episodes = []

    def _convert(self, obj):
        """Recursively convert numpy types to Python native types."""
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

    def log_episode(self, episode_idx, steps, total_reward, final_state):
        self.episodes.append({
            "episode": episode_idx,
            "steps": steps,
            "total_reward": total_reward,
            "final_state": final_state
        })

    def save(self):
        clean_episodes = self._convert(self.episodes)
        with open(self.log_path, "w") as f:
            json.dump(clean_episodes, f, indent=2)
        print(f"Logs saved to {self.log_path}")
