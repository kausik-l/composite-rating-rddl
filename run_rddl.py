import json
from env.rating_env import RatingEnv
from planner.policy import RandomPolicy, FixedPolicy, GreedyPolicy
from utils.logger import Logger
from utils.plotter import plot_results

def main():
    # ----------------------------------------------------------------
    # Point to the domain and instance files (RDDL definitions).
    #    - domain_file: defines the rules of the world.
    #    - instance_file: defines the objects (t1, English, French),
    #                     initial state, and horizon.
    # ----------------------------------------------------------------
    domain_file = "domains/sentiment_translation_domain.rddl"
    instance_file = "instances/sentiment_translation_instance1.rddl"

    # ----------------------------------------------------------------
    # Point to the datasets (CSV files).
    #    These datasets correspond to different pipeline plans:
    #      - english.csv: direct English sentiment analysis
    #      - french.csv: direct French sentiment analysis
    #      - roundtrip.csv: EN->FR->EN + English sentiment
    # ----------------------------------------------------------------
    data_files = {
        "english": "data/input/english.csv",
        "french": "data/input/french.csv",
        "roundtrip": "data/input/roundtrip.csv"
    }

    # ----------------------------------------------------------------
    # Create the environment.
    #    RatingEnv = RDDLEnv + Override reward with ARC metrics.
    #     - subset_size=100: evaluate each plan on a random subset of 100 rows
    # ----------------------------------------------------------------
    env = RatingEnv(domain_file, instance_file, data_files, subset_size=100)

    # ----------------------------------------------------------------
    # Define how high-level plans map to the actions.
    #    Plans = pipeline we choosr (english, french, roundtrip).
    #    Each plan is converted into the actual RDDL action.
    # ----------------------------------------------------------------
    plan_to_actions = {
        "english": ["do_sentiment_english___t1"],
        "french": ["do_sentiment_french___t1"],
        "roundtrip": [
            "do_translate___t1__English__French",
            "do_translate___t1__French__English",
            "do_sentiment_english___t1"
        ]
    }

    # ----------------------------------------------------------------
    # Choose a policy.
    #    - RandomPolicy: randomly picks english/french/roundtrip each episode.
    #    - FixedPolicy: always picks the same plan.
    #    - GreedyPolicy: Simulates each plan, picks the one with the best reward.
    # ----------------------------------------------------------------
    policy = RandomPolicy(env, plan_to_actions)

    # ----------------------------------------------------------------
    # Run multiple episodes.
    #    Each episode = pick a plan, run one action, sample dataset subset,
    #    compute WRS, log results and plot them.
    # ----------------------------------------------------------------
    n_episodes = 100
    results = []
    logger = Logger(log_dir="data/output", log_file="simulation_log.json")

    for ep in range(n_episodes):
        # Reset environment to the initial state (text in English, no sentiment yet).
        state = env.reset()
        print("\nState:",state)

        # Policy chooses one plan (english/french/roundtrip).
        chosen_plan = policy.select_plan(state)

        # Convert plan -> RDDL action dictionary.
        action_dict = policy.build_action_dict(chosen_plan)

        # Step the environment: 
        #   - Symbolic state update (RDDL)
        #   - ARC metrics computed on sampled dataset subset
        next_state, reward, done, truncated, info = env.step(action_dict)

        # Create a record of this episode for logging and analysis.
        record = {
            "episode": ep + 1,
            "plan": chosen_plan,
            "reward": reward,        # reward = -WRS 
            "wrs": info["wrs"],      # Weighted Rejection Score.
            "pie": info["pie"],      # placeholder.
            "ate": info["ate"]       # placeholder.
        }
        results.append(record)

        print(f"Episode {ep+1}: Plan={chosen_plan}, Reward={reward:.3f}, Metrics={info}")

        logger.log_episode(ep + 1, [record], total_reward=reward, final_state={})

    # ----------------------------------------------------------------
    # After all episodes:
    #    - Write logs to a JSON file.
    #    - Generate plots.
    # ----------------------------------------------------------------
    logger.save()
    plot_results("data/output/simulation_log.json", save_dir="simulation_plots")

if __name__ == "__main__":
    main()
