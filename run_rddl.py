import json
from env.rating_env import RatingEnv
from planner.policy import RandomPolicy, FixedPolicy, GreedyPolicy
from utils.logger import Logger
from utils.plotter import plot_results

def main():
    domain_file = "domains/sentiment_translation_domain.rddl"
    instance_file = "instances/sentiment_translation_instance1.rddl"
    data_files = {
        "english": "data/input/english.csv",
        "french": "data/input/french.csv",
        "roundtrip": "data/input/roundtrip.csv"
    }

    env = RatingEnv(domain_file, instance_file, data_files, subset_size=100)

    plan_to_actions = {
        "english": ["do_sentiment_english___t1"],
        "french": ["do_sentiment_french___t1"],
        "roundtrip": [
            "do_translate___t1__English__French",
            "do_translate___t1__French__English",
            "do_sentiment_english___t1"
        ]
    }

    # Pick a policy
    policy = RandomPolicy(env, plan_to_actions)

    n_episodes = 100
    results = []
    logger = Logger(log_dir="data/output", log_file="simulation_log.json")

    for ep in range(n_episodes):
        state = env.reset()
        chosen_plan = policy.select_plan(state)
        action_dict = policy.build_action_dict(chosen_plan)

        next_state, reward, done, truncated, info = env.step(action_dict)

        record = {
            "episode": ep + 1,
            "plan": chosen_plan,
            "reward": reward,
            "wrs": info["wrs"],
            "pie": info["pie"],
            "ate": info["ate"]
        }
        results.append(record)
        print(f"Episode {ep+1}: Plan={chosen_plan}, Reward={reward:.3f}, Metrics={info}")

        # Log to JSON using your Logger
        logger.log_episode(ep + 1, [record], total_reward=reward, final_state={})


    # Save logs
    logger.save()

    # Generate plots
    plot_results("data/output/simulation_log.json", save_dir="simulation_plots")

if __name__ == "__main__":
    main()
