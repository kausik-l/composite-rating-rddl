import os
import numpy as np
import pandas as pd

def generate_large_scenario(num_stages, num_models_per_stage, num_samples=1000, output_dir="."):
    """
    This function creates everything needed for our simulation run.
    
    Inputs:
        num_stages: How long is the pipeline?
        num_models_per_stage: How many choices do we have at each step? 
        num_samples: How many data points (rows) to generate? (I set it to 1000 here)
    
    Outputs:
        Saves:
        .csv file containing the synthetic data.
        .rddl domain file defining the rules.
        .rddl instance file defining the specific stages and models.
    """
    
    # DATA GENERATION
    # We create a synthetic dataset where model outputs depend on previous stages.
    print(f"Generating Data for {num_stages} stages x {num_models_per_stage} models...")
    
    # TODO (Add more Zs): Create Protected Attributes (Z1, Z2, Z3)
    # These represent demographic features (e.g., Age, Gender, Race).
    # We use different probabilities (p=[...]) to simulate imbalance.
    Z1 = np.random.choice([0, 1, 2], size=num_samples, p=[0.45, 0.45, 0.10])
    Z2 = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
    Z3 = np.random.choice([0, 1, 2], size=num_samples, p=[0.3, 0.5, 0.2])
    
    data = { 'Z1': Z1, 'Z2': Z2, 'Z3': Z3 }
    
    # The pipeline processes a signal. We start by combining the protected attributes
    # to create a base score (normalized roughly between 0 and 1).
    prev_stage_signal = (Z1 + Z2 + Z3) / 5.0 
    
    col_names = []
    
    # Simulate the Pipeline Loop
    # We loop through every stage (s) and every model (m) to generate data columns.
    for s in range(1, num_stages + 1):
        stage_cols = []
        for m in range(1, num_models_per_stage + 1):
            # Naming Convention: s{Stage}_m{Model} (e.g., s1_m1, s10_m5)
            col_name = f"s{s}_m{m}"
            stage_cols.append(col_name)
            
            # Each model adds a random "bias" and random "noise".
            bias = np.random.uniform(-0.2, 0.2)
            noise = np.random.normal(0, 0.1, num_samples)
            
            # Calculate raw value: Previous Signal + Bias + Noise
            vals = prev_stage_signal + bias + noise
            
            # If we just kept adding numbers for 100 stages, values would hit infinity.
            # So we use a Sigmoid function to squash the result between 0.0 and 5.0.
            vals = 5.0 / (1.0 + np.exp(-(vals - 0.5) * 2))
            
            data[col_name] = np.round(vals, 2)
        
        # The input for the next stage is the average output of the current stage.
        # We normalize it back to a small range.
        stage_avg = np.mean([data[c] for c in stage_cols], axis=0)
        prev_stage_signal = (stage_avg / 5.0)

    # Save the DataFrame to CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "data", "input", f"chain_{num_stages}_{num_models_per_stage}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  -> Data saved to {csv_path}")

    # PART 2: RDDL DOMAIN GENERATION 
    # This defines the physics of the world. It doesn't change much based on size.
    domain_name = f"chain_domain_{num_stages}_{num_models_per_stage}"
    
    # We use a Python f-string to insert variables into the RDDL code block.
    domain_content = f"""domain {domain_name} {{
    types {{
        stage : object;
        model : object;
    }};

    pvariables {{
        // Non-Fluents (Static facts about the map)
        valid_model(stage, model) : {{ non-fluent, bool, default = false }};
        next_stage(stage, stage)  : {{ non-fluent, bool, default = false }};
        is_start_stage(stage)     : {{ non-fluent, bool, default = false }};
        is_final_stage(stage)     : {{ non-fluent, bool, default = false }};
        model_call_cost           : {{ non-fluent, real, default = 0.05 }};

        // State Fluents (Variables that change during the episode)
        current_stage(stage)      : {{ state-fluent, bool, default = false }};
        pipeline_done             : {{ state-fluent, bool, default = false }};

        // Action Fluents (What the agent controls)
        select_model(model)       : {{ action-fluent, bool, default = false }};
    }};

    cpfs {{
        // Transition Logic: How 'current_stage' updates
        current_stage'(?next) = 
            if (pipeline_done) then false
            // If we are at ?curr, pick ?m, and ?curr leads to ?next... move to ?next
            else if (exists_{{?curr : stage, ?m : model}} [
                current_stage(?curr) ^ select_model(?m) ^ next_stage(?curr, ?next)
            ]) then true
            // If we didn't pick an action, stay where we are
            else if (current_stage(?next) ^ ~(exists_{{?m : model}} [select_model(?m)])) then true
            else false; 

        // Termination Logic: We are done if we pick a model at the final stage
        pipeline_done' = 
            pipeline_done | 
            exists_{{?s : stage, ?m : model}} [
                current_stage(?s) ^ is_final_stage(?s) ^ select_model(?m)
            ];
    }};

    // Preconditions: You can only pick a model valid for the current stage
    action-preconditions {{
        forall_{{?m : model}} [
            select_model(?m) => exists_{{?s : stage}} [current_stage(?s) ^ valid_model(?s, ?m)]
        ];
    }};

    // Reward: Simple cost per step (-5.0). The Fairness penalty is added in Python.
    reward = sum_{{?m : model}} [ -model_call_cost * select_model(?m) ];
}}
"""
    domain_path = os.path.join(output_dir, "domains", f"chain_{num_stages}_{num_models_per_stage}.rddl")
    os.makedirs(os.path.dirname(domain_path), exist_ok=True)
    with open(domain_path, "w") as f:
        f.write(domain_content)
    print(f"  -> Domain saved to {domain_path}")

    # PART 3: RDDL INSTANCE GENERATION 
    # This defines the specific topology (100 stages) and objects (20 models).
    instance_name = f"inst_{num_stages}_{num_models_per_stage}"
    
    # Stages are unique (s1, s2... s100)
    stage_objs = [f"s{i}" for i in range(1, num_stages + 1)]
    
    # Models are REUSED (m1...m20). 
    # This prevents creating 2000 objects, which crashes the RDDL compiler.
    model_objs = [f"m{i}" for i in range(1, num_models_per_stage + 1)]

    # Define Topology (s1 -> s2 -> s3...)
    nf_logic = []
    nf_logic.append(f"is_start_stage({stage_objs[0]});")
    nf_logic.append(f"is_final_stage({stage_objs[-1]});")
    
    for i in range(len(stage_objs) - 1):
        nf_logic.append(f"next_stage({stage_objs[i]}, {stage_objs[i+1]});")
        
    # Every model (m1..m20) is valid at every stage.
    # The Python Environment handles mapping (Stage 5 + Model 1) -> (Column s5_m1).
    for s_obj in stage_objs:
        for m_obj in model_objs:
            nf_logic.append(f"valid_model({s_obj}, {m_obj});")

    instance_content = f"""non-fluents nf_{instance_name} {{
    domain = {domain_name};
    objects {{
        stage : {{ {", ".join(stage_objs)} }};
        model : {{ {", ".join(model_objs)} }};
    }};
    non-fluents {{
        model_call_cost = 5.0;
        {chr(10).join(nf_logic)}
    }};
}}

instance {instance_name} {{
    domain = {domain_name};
    non-fluents = nf_{instance_name};
    init-state {{
        current_stage({stage_objs[0]});
        pipeline_done = false;
    }};
    max-nondef-actions = 1;
    horizon = {num_stages + 5}; // Horizon is stages + buffer
    discount = 1.0;
}}
"""
    instance_path = os.path.join(output_dir, "instances", f"chain_{num_stages}_{num_models_per_stage}_inst.rddl")
    os.makedirs(os.path.dirname(instance_path), exist_ok=True)
    with open(instance_path, "w") as f:
        f.write(instance_content)
    print(f"  -> Instance saved to {instance_path}")
    
    return domain_path, instance_path, csv_path

if __name__ == "__main__":
    # Example: 5 stages, 3 models each
    generate_large_scenario(5, 3)