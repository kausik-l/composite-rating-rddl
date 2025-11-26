import os
import numpy as np
import pandas as pd

def generate_large_scenario(num_stages, num_models_per_stage, num_families=2, num_samples=1000, output_dir="."):
    """
    The World Factory: Generates the simulation environment.
    
    This function creates three essential components for the experiment:
    1. Synthetic Data (CSV): Represents a population moving through a pipeline. 
       Includes structural bias where one model family is less fair than the other.
    2. RDDL Domain: Defines the 'Physics' (costs, transitions, states).
    3. RDDL Instance: Defines the 'Map' (specific stages and model families).
    
    Args:
        num_stages: Length of the pipeline (e.g., 20 steps).
        num_models_per_stage: Choices available at each step (e.g., 6 models).
        num_families: Groupings of models (e.g., 2 Families). Switching families incurs a cost.
    """
    
    print(f"Generating Scenario: {num_stages} Stages, {num_models_per_stage} Models, {num_families} Families...")
    
    # =========================================================================
    # PART 1: DATA GENERATION (The Causal Chain)
    # We simulate a process where 'True Merit' (T) and 'Protected Attribute' (Z)
    # both influence the final outcome (Y).
    
    # 1.1 Generate Protected Attribute (Z)
    # Z1 is our primary protected attribute (e.g., Gender or Race).
    # Z2/Z3 are generated as noise to test robustness.
    Z = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
    Z2_noise = np.random.choice([0, 1], size=num_samples)
    Z3_noise = np.random.choice([0, 1, 2], size=num_samples)
    
    # 1.2 Generate True Merit (T) with Systemic Bias
    # In many real-world scenarios, 'Merit' (e.g., education, history) is 
    # already correlated with Z due to systemic factors.
    # Here, Group 1 (Z=1) has a slight statistical advantage in T.
    T_noise = np.random.normal(0, 0.1, num_samples)
    T = 0.4 + (0.2 * Z) + T_noise
    T = np.clip(T, 0.0, 1.0)
    
    # Prepare the DataFrame
    data = { 'Z1': Z, 'Z2': Z2_noise, 'Z3': Z3_noise, 'T': T }
    
    # The 'Signal' flowing through the pipeline starts as True Merit.
    prev_stage_signal = T.copy()
    
    # 1.3 Generate Model Outputs for every Stage
    for s in range(1, num_stages + 1):
        stage_cols = []
        for m in range(1, num_models_per_stage + 1):
            col_name = f"s{s}_m{m}"
            stage_cols.append(col_name)
            
            # Determine Family (Round Robin assignment)
            # Model 1 -> Family 1, Model 2 -> Family 2, etc.
            fam_idx = (m - 1) % num_families
            
            # --- STRUCTURAL BIAS LOGIC ---
            # We intentionally make Family 1 biased and Family 2 fair.
            # The agent must learn this distinction.
            
            # Alpha: Sensitivity to Merit (Good). All models care about T.
            alpha = np.random.uniform(0.8, 1.2)
            
            # Beta: Sensitivity to Protected Attribute (Bad).
            if fam_idx == 0: 
                # Family 1 (Biased): Explicitly uses Z to score people.
                beta = np.random.uniform(0.1, 0.3)
            else:
                # Family 2 (Fair): Mostly ignores Z.
                beta = np.random.uniform(-0.05, 0.05)
            
            # Calculate Output Y
            noise = np.random.normal(0, 0.05, num_samples)
            raw_y = (alpha * prev_stage_signal) + (beta * Z) + noise
            
            # Normalize to 0-5 range (Sigmoid) to keep values stable across stages
            y_scaled = 5.0 / (1.0 + np.exp(-(raw_y - 0.5) * 4))
            data[col_name] = np.round(y_scaled, 2)
        
        # Propagate Signal: The average output of Stage S becomes input for Stage S+1
        stage_avg = np.mean([data[c] for c in stage_cols], axis=0)
        prev_stage_signal = stage_avg / 5.0

    # Save Data to Disk
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "data", "input", f"chain_{num_stages}_{num_models_per_stage}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  -> Data saved to {csv_path}")

    # =========================================================================
    # PART 2: RDDL DOMAIN GENERATION (The Rules)
    # Defines costs and state transitions.
    domain_name = f"chain_family_{num_stages}_{num_models_per_stage}"
    
    domain_content = f"""domain {domain_name} {{
    types {{
        stage : object;
        model : object;
        family : object; 
    }};

    pvariables {{
        // Topology Definitions
        valid_model(stage, model) : {{ non-fluent, bool, default = false }};
        next_stage(stage, stage)  : {{ non-fluent, bool, default = false }};
        is_start_stage(stage)     : {{ non-fluent, bool, default = false }};
        is_final_stage(stage)     : {{ non-fluent, bool, default = false }};
        is_in_family(model, family) : {{ non-fluent, bool, default = false }};
        
        // Cost Configuration
        // We set costs low (0.5) to allow the agent to make trade-offs easily.
        base_cost      : {{ non-fluent, real, default = 0.5 }};
        switching_cost : {{ non-fluent, real, default = 0.5 }}; 

        // State Tracking
        current_stage(stage)      : {{ state-fluent, bool, default = false }};
        last_used_family(family)  : {{ state-fluent, bool, default = false }};
        pipeline_done             : {{ state-fluent, bool, default = false }};
        
        // Actions
        select_model(model)       : {{ action-fluent, bool, default = false }};
    }};

    cpfs {{
        // Transition: Move to next stage if valid move taken
        current_stage'(?next) = 
            if (pipeline_done) then false
            else if (exists_{{?curr : stage, ?m : model}} [
                current_stage(?curr) ^ select_model(?m) ^ next_stage(?curr, ?next)
            ]) then true
            else if (current_stage(?next) ^ ~(exists_{{?m : model}} [select_model(?m)])) then true
            else false; 

        // Memory: Track which family was just used
        last_used_family'(?next_fam) = 
            if (exists_{{?m : model}} [ select_model(?m) ^ is_in_family(?m, ?next_fam) ])
            then true
            else if (~(exists_{{?m : model}} [select_model(?m)]) ^ last_used_family(?next_fam))
            then true
            else false;

        pipeline_done' = 
            pipeline_done | 
            exists_{{?s : stage, ?m : model}} [
                current_stage(?s) ^ is_final_stage(?s) ^ select_model(?m)
            ];
    }};

    action-preconditions {{
        forall_{{?m : model}} [
            select_model(?m) => exists_{{?s : stage}} [current_stage(?s) ^ valid_model(?s, ?m)]
        ];
    }};

    // Reward Function: Base Cost + Switching Cost
    // Fairness penalty is added externally by the Python Environment.
    reward = 
        (sum_{{?m : model}} [ -base_cost * select_model(?m) ]) +
        (sum_{{?m : model, ?new_fam : family, ?old_fam : family}} [
            if (select_model(?m) ^ is_in_family(?m, ?new_fam) ^ last_used_family(?old_fam) ^ (?new_fam ~= ?old_fam))
            then -switching_cost
            else 0.0
        ]);
}}
"""
    domain_path = os.path.join(output_dir, "domains", f"chain_family_{num_stages}_{num_models_per_stage}.rddl")
    os.makedirs(os.path.dirname(domain_path), exist_ok=True)
    with open(domain_path, "w") as f:
        f.write(domain_content)
    print(f"  -> Domain saved to {domain_path}")

    # =========================================================================
    # PART 3: RDDL INSTANCE GENERATION (The Map)
    # Defines the specific objects and their relationships.
    instance_name = f"inst_family_{num_stages}_{num_models_per_stage}"
    
    stage_objs = [f"s{i}" for i in range(1, num_stages + 1)]
    model_objs = [f"m{i}" for i in range(1, num_models_per_stage + 1)]
    family_objs = [f"fam_{i}" for i in range(1, num_families + 1)]
    family_objs.append("fam_none") # Initial state (no family used yet)

    nf_logic = []
    
    # Define Sequence (s1 -> s2 -> s3...)
    nf_logic.append(f"is_start_stage({stage_objs[0]});")
    nf_logic.append(f"is_final_stage({stage_objs[-1]});")
    for i in range(len(stage_objs) - 1):
        nf_logic.append(f"next_stage({stage_objs[i]}, {stage_objs[i+1]});")
        
    # Define Validity (All models valid at all stages)
    for s_obj in stage_objs:
        for m_obj in model_objs:
            nf_logic.append(f"valid_model({s_obj}, {m_obj});")
            
    # Define Family Assignments (Round Robin)
    for i, m_obj in enumerate(model_objs):
        fam_index = (i % num_families) + 1
        fam_obj = f"fam_{fam_index}"
        nf_logic.append(f"is_in_family({m_obj}, {fam_obj});")

    instance_content = f"""non-fluents nf_{instance_name} {{
    domain = {domain_name};
    objects {{
        stage : {{ {", ".join(stage_objs)} }};
        model : {{ {", ".join(model_objs)} }};
        family: {{ {", ".join(family_objs)} }};
    }};
    non-fluents {{
        base_cost = 0.5;
        switching_cost = 0.5; 
        {chr(10).join(nf_logic)}
    }};
}}

instance {instance_name} {{
    domain = {domain_name};
    non-fluents = nf_{instance_name};
    init-state {{
        current_stage({stage_objs[0]});
        last_used_family(fam_none);
        pipeline_done = false;
    }};
    max-nondef-actions = 1;
    horizon = {num_stages + 5}; 
    discount = 1.0;
}}
"""
    instance_path = os.path.join(output_dir, "instances", f"chain_family_{num_stages}_{num_models_per_stage}_inst.rddl")
    os.makedirs(os.path.dirname(instance_path), exist_ok=True)
    with open(instance_path, "w") as f:
        f.write(instance_content)
    print(f"  -> Instance saved to {instance_path}")
    
    return domain_path, instance_path, csv_path

if __name__ == "__main__":
    generate_large_scenario(10, 5, 2)