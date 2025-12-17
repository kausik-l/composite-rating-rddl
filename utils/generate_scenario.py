import os
import numpy as np
import pandas as pd

def generate_large_scenario(num_stages, num_models_per_stage, num_families=2, num_samples=1000, output_dir="."):
    """
    Generates a complete RDDL domain/instance and Causal Data.
    
    Args:
        num_stages: Number of sequential stages.
        num_models_per_stage: Number of choices per stage.
        num_families: Number of 'lanes'.
        num_samples: Rows of synthetic data (Crucial for statistical stability).
    """
    
    print(f"Generating Causal Scenario: {num_stages} Stages, {num_models_per_stage} Models, {num_samples} Samples...")
    
    # =========================================================================
    # PART 1: CAUSAL DATA GENERATION
    # =========================================================================
    
    # 1.1 Generate Protected Attribute (Z) using num_samples
    Z = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
    Z2_noise = np.random.choice([0, 1], size=num_samples)
    Z3_noise = np.random.choice([0, 1, 2], size=num_samples)
    
    # 1.2 Generate True Merit (T) using num_samples
    T_noise = np.random.normal(0, 0.1, num_samples)
    T = 0.4 + (0.2 * Z) + T_noise
    T = np.clip(T, 0.0, 1.0)
    
    data = { 'Z1': Z, 'Z2': Z2_noise, 'Z3': Z3_noise, 'T': T }
    prev_stage_signal = T.copy()
    
    # 1.3 Generate Model Outputs
    for s in range(1, num_stages + 1):
        stage_cols = []
        for m in range(1, num_models_per_stage + 1):
            col_name = f"s{s}_m{m}"
            stage_cols.append(col_name)
            
            # Determine Family
            fam_idx = (m - 1) % num_families
            
            # Bias Logic
            alpha = np.random.uniform(0.8, 1.2)
            if fam_idx == 0: 
                beta = np.random.uniform(0.1, 0.3)
            else:
                beta = np.random.uniform(-0.05, 0.05)
            
            # Generate Y using num_samples
            noise = np.random.normal(0, 0.05, num_samples)
            raw_y = (alpha * prev_stage_signal) + (beta * Z) + noise
            
            y_scaled = 5.0 / (1.0 + np.exp(-(raw_y - 0.5) * 4))
            data[col_name] = np.round(y_scaled, 2)
        
        stage_avg = np.mean([data[c] for c in stage_cols], axis=0)
        prev_stage_signal = stage_avg / 5.0

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "data", "input", f"chain_{num_stages}_{num_models_per_stage}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  -> Data saved to {csv_path}")

    # =========================================================================
    # PART 2 & 3: RDDL GENERATION (Standard)
    # =========================================================================
    domain_name = f"chain_family_{num_stages}_{num_models_per_stage}"
    
    domain_content = f"""domain {domain_name} {{
    types {{
        stage : object;
        model : object;
        family : object; 
    }};

    pvariables {{
        valid_model(stage, model) : {{ non-fluent, bool, default = false }};
        next_stage(stage, stage)  : {{ non-fluent, bool, default = false }};
        is_start_stage(stage)     : {{ non-fluent, bool, default = false }};
        is_final_stage(stage)     : {{ non-fluent, bool, default = false }};
        is_in_family(model, family) : {{ non-fluent, bool, default = false }};
        
        base_cost      : {{ non-fluent, real, default = 0.5 }};
        switching_cost : {{ non-fluent, real, default = 0.5 }}; 

        current_stage(stage)      : {{ state-fluent, bool, default = false }};
        last_used_family(family)  : {{ state-fluent, bool, default = false }};
        pipeline_done             : {{ state-fluent, bool, default = false }};
        select_model(model)       : {{ action-fluent, bool, default = false }};
    }};

    cpfs {{
        current_stage'(?next) = 
            if (pipeline_done) then false
            else if (exists_{{?curr : stage, ?m : model}} [
                current_stage(?curr) ^ select_model(?m) ^ next_stage(?curr, ?next)
            ]) then true
            else if (current_stage(?next) ^ ~(exists_{{?m : model}} [select_model(?m)])) then true
            else false; 

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

    # Instance Generation
    instance_name = f"inst_family_{num_stages}_{num_models_per_stage}"
    
    stage_objs = [f"s{i}" for i in range(1, num_stages + 1)]
    model_objs = [f"m{i}" for i in range(1, num_models_per_stage + 1)]
    family_objs = [f"fam_{i}" for i in range(1, num_families + 1)]
    family_objs.append("fam_none")

    nf_logic = []
    nf_logic.append(f"is_start_stage({stage_objs[0]});")
    nf_logic.append(f"is_final_stage({stage_objs[-1]});")
    
    for i in range(len(stage_objs) - 1):
        nf_logic.append(f"next_stage({stage_objs[i]}, {stage_objs[i+1]});")
    for s_obj in stage_objs:
        for m_obj in model_objs:
            nf_logic.append(f"valid_model({s_obj}, {m_obj});")
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
    generate_large_scenario(10, 5, 2, num_samples=30000)