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
├── domains # Contains the RDDL files for all three tasks. 
│
├── planner/             # Planning and policy implementations
│   ├── policy.py
│   └── baselines.py
│
├── data/          # Input datasets for synthetic experiments
│
├── results/             # Experiment outputs and visual plots
│
├── run_sentiment_small.py # Main experiment runners
├── run_sentiment_large.py
├── rating_env.py      
└── README.md
```

## Tasks Implemented

The repository includes implementations for the three primary tasks discussed in the paper:

### T1 – RTS-Small
A two-stage composite model: **Translation → Sentiment Analysis**.  
This task evaluates robustness under statistical bias without confounding factors.  
* **Environment:** `env/sentiment_small_env.py`

### T2 – RTS-Large
A larger sentiment pipeline utilizing the **ALLURE dataset**.  
This introduces confounding effects between protected attributes and treatment variables.  
* **Environment:** `env/sentiment_large_env.py`

### T3 – Synthetic Chain
A synthetic environment modeling deep composite AI pipelines with many stages.  
* **Key Features:** Multiple model families, switching costs, and algorithmic bias.  
* **Environment:** `env/dynamic_chain_env.py`



---

## Core Components

### 1. Environment
All tasks are implemented as **Markov Decision Process (MDP)** environments where:
* **State:** Represents pipeline progress and current context.
* **Action:** Selecting a specific primitive model for the current stage.
* **Reward:** A weighted combination of robustness metrics and model invocation cost.
* **Abstraction:** `rating_env.py`



### 2. Planner
The planner utilizes tabular **Q-learning** to derive selection policies.
* **Policies:** Found in `planner/policy.py` and `planner/sentiment_policy.py`.
* **Baselines:** Includes random selection, fixed model choices, and heuristic lookahead (implemented in `planner/baselines.py`).

---

## Getting Started

### Installation
Clone this repo and install the required dependencies using pip:

```
pip install -r requirements.txt
```

## Running Experiments
You can run the experiments for specific tasks using the provided scripts:
1. Run RTS-Small: ```python run_sentiment_small.py```
2. Run RTS-Large: ```python run_sentiment_large.py```
3. Run Synthetic Chain: ```python run_family_experiment.py```


## Citation
If you use this code or our research in your work, please cite:
```
@inproceedings{lakkaraju2026composite,
  title={Assessing the Robustness of Composite AI Models via Probabilistic Planning},
  author={Lakkaraju, Kausik and Patra, Sunandita and Zehtabi, Parisa and Srivastava, Biplav},
  booktitle={ICAPS},
  year={2026}
}
```


