import pandas as pd
import numpy as np

def generate_dataset(n_samples, bias_rule, noise_std=0.1, seed=None):
    """
    Generate a synthetic dataset with treatments, protected attribute, and biased outcomes.

    Args:
        n_samples: number of rows
        bias_rule: function mapping gender -> base outcome
        noise_std: standard deviation of noise added to outcome
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    treatments = ["positive", "negative"]
    protected = ["male", "female", "neutral"]

    data = []
    for _ in range(n_samples):
        treatment = rng.choice(treatments)
        gender = rng.choice(protected)
        outcome = bias_rule(gender)

        # add noise (so that we get different metric results for each subset of the data). 
        outcome = outcome + rng.normal(0, noise_std)

        # keep in [-1,1]
        outcome = np.clip(outcome, -1, 1)  

        data.append([treatment, gender, outcome])

    return pd.DataFrame(data, columns=["treatment", "gender", "sentiment_outcome"])


def english_bias(gender):
    if gender == "male": return 1
    elif gender == "female": return -1
    else: return 0

def french_bias(gender):
    if gender == "male": return 1
    # female & neutral = -1
    else: return -1  

def roundtrip_bias(gender):
    # all positive.
    return 1  


if __name__ == "__main__":
    # Generate all datasets and save them.
    n_samples = 1000
    english = generate_dataset(n_samples, english_bias, noise_std=0.1, seed=42)
    french = generate_dataset(n_samples, french_bias, noise_std=0.1, seed=43)
    roundtrip = generate_dataset(n_samples, roundtrip_bias, noise_std=0.1, seed=44)

    english.to_csv("data/input/english.csv", index=False)
    french.to_csv("data/input/french.csv", index=False)
    roundtrip.to_csv("data/input/roundtrip.csv", index=False)

    print("Datasets generated and saved to data/input/")
