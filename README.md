# composite-rating-rddl
This repo contains an example RDDL-based simulation to quantify robustness in composite AI systems.

# Example-1: Sentiment Analysis System (SAS) + Translator

We model a system that needs to perform sentiment analysis on text. The text might not be in English, so the agent must decide:
- whether to run a general-purpose sentiment tool (which works on some pre-defined languages),
- whether to translate the text to English using a translator and use a English-only SAS.

Each action has a cost. The goal is to get maximum robustness.

## Files

- `domain.rddl`: Defines the RDDL domain with actions, state fluents, cost functions, and reward logic.
- `instance.rddl`: Sets up an example problem where a French text needs sentiment analysis.
- `policy.py`: A placeholder agent that blindly samples valid actions (yet to replace with a real policy).
- `run.py`: Runs the simulation and saves visualizations of each decision step.
- `plots/`: Output images showing how the system evolves over time.

## How to run

Install `pyRDDLGym`:

```bash
pip install pyRDDLGym
python run.py
