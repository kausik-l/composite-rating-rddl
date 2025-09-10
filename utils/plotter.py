import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def plot_results(log_file="data/output/simulation_log.json", save_dir="simulation_plots"):
    os.makedirs(save_dir, exist_ok=True)

    with open(log_file, "r") as f:
        data = json.load(f)

    # Flatten nested "steps"
    rows = []
    for ep in data:
        for step in ep["steps"]:
            row = {
                "episode": ep["episode"],
                "plan": step["plan"],
                "reward": step["reward"],
                "wrs": step["wrs"],
                "pie": step["pie"],
                "ate": step["ate"]
            }
            rows.append(row)

    df = pd.DataFrame(rows)


    # === 1. WRS distribution by plan ===
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="plan", y="wrs", data=df, palette="Set2")
    plt.title("Bias (WRS) Distribution Across Plans")
    plt.ylabel("Weighted Rejection Score (lower = less bias)")
    plt.xlabel("Plan")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/wrs_distribution.png")
    plt.close()

    # === 2. Average reward per plan ===
    plt.figure(figsize=(8, 5))
    sns.barplot(x="plan", y="reward", data=df, errorbar="sd", palette="Set1")
    plt.title("Average Reward Across Plans")
    plt.ylabel("Reward (higher = better)")
    plt.xlabel("Plan")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_reward.png")
    plt.close()

    # === 3. Reward trend over episodes ===
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="episode", y="reward", hue="plan", data=df, marker="o")
    plt.title("Reward Trend Over Episodes")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reward_trend.png")
    plt.close()

    print(f"Plots saved in {save_dir}")
