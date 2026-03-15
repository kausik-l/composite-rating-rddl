# Composite-Rating-RDDL

[![Paper](https://img.shields.io/badge/ICAPS-2026-blue)](https://icaps-conference.org/)
[![License](https://img.shields.io/badge/License-Research-green)](#license)

This repository contains the official implementation accompanying our ICAPS paper:  
**"Assessing the Robustness of Composite AI Models via Probabilistic Planning"**

The project investigates the selection of primitive models in multi-stage AI pipelines using probabilistic planning and reinforcement learning. By treating model selection as a sequential decision problem, the framework allows for the optimization of robustness metrics—specifically causal fairness measures—across complex composite workflows.

## Overview

Modern AI systems are frequently **composite models**—pipelines built from multiple primitive components (e.g., *Translation → Sentiment Analysis*).



The choice of primitive models at each stage significantly influences the overall system's robustness, bias, and operational cost. This repository implements a planning-based framework that:

1.  **Models Construction as an MDP:** Represents composite model building as a sequential decision process.
2.  **Learns Optimal Policies:** Uses Q-learning to determine the best primitive model for each stage.
3.  **Evaluates Robustness:** Uses causal metrics (ATE, WRS, DIE) to assess model pipelines.
4.  **Flexible Environments:** Supports both real-world sentiment analysis tasks and complex synthetic multi-stage chains.

---

## Repository Structure

Below are the core components of the project:

```text
Composite-Rating-RDDL
│
├── env/                 # Environment definitions (MDP/RDDL-style tasks)
│   ├── sentiment_small_env.py
│   ├── sentiment_large_env.py
│   └── dynamic_chain_env.py
│
├── planner/             # Planning and policy implementations
│   ├── policy.py
│   ├── sentiment_policy.py
│   └── baselines.py
│
├── data_input/          # Input datasets for synthetic experiments
│   └── real_world/
│
├── results/             # Experiment outputs and visual plots
│
├── run_experiments.py   # Main experiment runner
├── run_sentiment_small.py
├── run_sentiment_large.py
```


│
├── rating_env.py        # Core environment abstraction
└── README.md
