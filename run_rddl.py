import json
from env.rating_env import RatingEnv
from planner.policy import RandomPolicy, FixedPolicy, GreedyPolicy
from utils.logger import Logger
from utils.plotter import plot_results


# Build a plan dynamically based on the current state.
def build_plan(state, plan_name):
    in_english = state.get("text_in_lang___t1__English", False)
    in_french = state.get("text_in_lang___t1__French", False)

    if plan_name == "english":
        if in_english:
            return [["do_sentiment_english___t1"]]
        elif in_french:
            return [["do_translate___t1__French__English"],
                    ["do_sentiment_english___t1"]]

    if plan_name == "french":
        if in_french:
            return [["do_sentiment_french___t1"]]
        elif in_english:
            return [["do_translate___t1__English__French"],
                    ["do_sentiment_french___t1"]]

    if plan_name == "roundtrip":
        if in_english:
            return [["do_translate___t1__English__French"],
                    ["do_translate___t1__French__English"],
                    ["do_sentiment_english___t1"]]
        elif in_french:
            return [["do_translate___t1__French__English"],
                    ["do_translate___t1__English__French"],
                    ["do_sentiment_french___t1"]]

    raise ValueError(f"Unsupported plan {plan_name} for state {state}")


def main():
    # Load domain and instance files.
    domain_file = "domains/sentiment_translation_domain.rddl"
    instance_file = "instances/sentiment_translation_instance1.rddl"

    # Data files for each plan with predictions.
    data_files = {
        "english": "data/input/english.csv",
        "french": "data/input/french.csv",
        "roundtrip": "data/input/roundtrip.csv"
    }

    # Environment (RDDL + Python wrapper for metrics).
    env = RatingEnv(domain_file, instance_file, data_files, subset_size=100, metric_weights={"wrs": 1.0, "action_cost": 10})


    # Choosing a policy.
    policy = RandomPolicy(env, ["english", "french", "roundtrip"])

    n_episodes = 50
    logger = Logger(log_dir="data/output", log_file="simulation_log.json")

    for ep in range(n_episodes):
        state, _ = env.reset()   # unpack: state_fluents, non_fluents
        chosen_plan = policy.select_plan(state)

        plan_steps = build_plan(state, chosen_plan)


        total_reward = 0
        episode_steps = []

        print(f"\n=== Episode {ep+1}: Executing plan {chosen_plan} ===")

        for step_idx, step_actions in enumerate(plan_steps, start=1):
            action_dict = {name: 0 for name in env.action_space.spaces.keys()}
            for act in step_actions:
                action_dict[act] = 1

            next_state, reward, done, truncated, info = env.step(action_dict)
            # unpack new state for next iteration
            state = next_state  

            total_reward += reward

            # Record this step
            step_record = {
                "episode": ep + 1,
                "plan": chosen_plan,
                "step_idx": step_idx,
                "actions": step_actions,
                "reward": reward,
                "wrs": info.get("wrs", 0.0),
                "pie": info.get("pie", 0.0),
                "ate": info.get("ate", 0.0),
            }
            episode_steps.append(step_record)

            print(f" Step {step_idx}: {step_actions}, Reward={reward:.3f}, State={state}")

            if done or truncated:
                break

        # Record the full episode
        record = {
            "episode": ep + 1,
            "steps": episode_steps,
            "total_reward": total_reward,
            "final_state": state
        }
        logger.episodes.append(record)
        print(f"Episode {ep+1} finished. Total Reward={total_reward:.3f}")

    # Save logs and generate plots.
    logger.save()
    plot_results("data/output/simulation_log.json", save_dir="simulation_plots")


if __name__ == "__main__":
    main()
