from pyRDDLGym.core.env import RDDLEnv
from policy import build_policy
import matplotlib.pyplot as plt

def main():
    env = RDDLEnv(domain='domain.rddl', instance='instance.rddl')
    policy = build_policy(env)
    obs, _ = env.reset()
    total_reward = 0

    for step in range(env.horizon):
        # Get the environment-rendered image
        img = env.render()

        # Save it directly using matplotlib without showing
        plt.figure(figsize=(6, 4))  # Optional: adjust as needed
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{step}.png", bbox_inches='tight', pad_inches=0)
        plt.close()  # Ensures nothing is shown or kept in memory

        # Take action
        action = policy.sample_action(obs)
        result = env.step(action)
        obs, reward, done, violated, info = result

        print(f"Step {step}, Action: {action}, Reward: {reward:.2f}")
        total_reward += reward

        if done:
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
