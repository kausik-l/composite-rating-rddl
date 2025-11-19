#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from env.rating_env import RatingEnv
from planner.policy import SimplePolicy  # use the small testing policy

class SimpleLogger:
    def __init__(self, out_dir: str, fname: str = "run_log.json"):
        self.path = Path(out_dir) / fname
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"episodes": []}

    def add_episode(self, record):
        self.data["episodes"].append(record)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Saved results to {self.path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--subset_size", type=int, default=100)
    p.add_argument("--out_dir", default="results/run1")
    return p.parse_args()


def load_data_mapping(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".json":
        return json.load(open(p))
    return {"default": str(p)}


def safe_reset(env):
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r


def main():
    args = parse_args()
    data_map = load_data_mapping(args.data)

    env = RatingEnv(
        domain_file=args.domain,
        instance_file=args.instance,
        data_files=data_map,
        subset_size=args.subset_size,
    )

    policy = SimplePolicy(env)
    logger = SimpleLogger(args.out_dir)

    for ep in range(1, args.episodes + 1):
        state = safe_reset(env)
        done, total_reward = False, 0.0
        steps = []

        print(f"\n=== Episode {ep} ===")
        print("Initial state keys:", list(state.keys())[:10])

        for step in range(50):
            action = policy.get_action(state)
            if not action:
                print(f"Step {step}: no valid action, stopping.")
                break

            print(f"Step {step}: action -> {list(action.keys())}")
            next_state, reward, done, truncated, info = env.step(action)
            steps.append({"step": step, "action": action, "reward": reward, "info": info})
            total_reward += float(reward or 0.0)
            state = next_state
            if done or truncated:
                break

        logger.add_episode({
            "episode": ep,
            "steps": steps,
            "total_reward": total_reward
        })
        print(f"Episode {ep} finished. Total reward = {total_reward:.3f}, steps = {len(steps)}")

    logger.save()


if __name__ == "__main__":
    main()
