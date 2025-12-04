import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuration to match your saved files
CONFIG = {
    "NUM_STAGES": 10,
    "AGENTS_DIR": "../saved_agents"
}

def load_agent_history(mode_name):
    filename = f"q_agent_{mode_name}_{CONFIG['NUM_STAGES']}_stages.pkl"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG["AGENTS_DIR"], filename)
    
    if not os.path.exists(filepath):
        print(f"[WARNING] File not found: {filepath}")
        return []
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    # Check for history key
    if "history" in data:
        return data["history"]
    else:
        print(f"[WARNING] No 'history' key found in {filename}. Was it trained with the latest code?")
        return []

def plot_training_curves():
    print("Generating Training Plots from Saved Pickles...")
    
    modes = [
        ("WRS", "purple", "Q-Learning (WRS)"),
        ("DIE", "blue", "Q-Learning (DIE)"),
        ("BOTH", "green", "Q-Learning (Combined)")
    ]
    
    plt.figure(figsize=(12, 6))
    has_data = False
    
    for mode, color, label in modes:
        history = load_agent_history(mode)
        
        if history:
            has_data = True
            # Apply smoothing
            window = 50
            if len(history) > window:
                smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
                plt.plot(smoothed, color=color, label=label, linewidth=2)
            else:
                plt.plot(history, color=color, label=label, alpha=0.5)
                
            print(f"  > Plotting {mode}: {len(history)} episodes found.")
        else:
            print(f"  > Skipping {mode} (No data).")

    if has_data:
        plt.title(f"Training Convergence ({CONFIG['NUM_STAGES']} Stages)")
        plt.xlabel("Training Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        outfile = "plot_ablation_learning_restored.png"
        plt.savefig(outfile)
        print(f"\nSuccess! Saved plot to {outfile}")
        # plt.show() # Uncomment to view
    else:
        print("\n[ERROR] No training history found in any pickle files.")
        print("Please ensure you ran with --train at least once using the NEW code that saves history.")

if __name__ == "__main__":
    plot_training_curves()