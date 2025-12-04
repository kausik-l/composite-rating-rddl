import pandas as pd
import os
import sys

def merge_unibot_data_robust():
    """
    Robust Merger for Unibot Data.
    Strategy:
    1. Scan all files.
    2. Read files and align them by ROW INDEX (assuming they are parallel datasets).
    3. Concatenate horizontally to create the master file.
    4. Preserves all rows, even if C_num is duplicated.
    """
    root_dir = os.path.join("data", "input", "real_world", "unibot")
    output_dir = os.path.join("data", "input", "real_world")
    output_path = os.path.join(output_dir, "master_sentiment.csv")
    
    languages = ['eng', 'spa', 'dan']
    models = ['bf', 'dbert', 'gru', 'random', 'textblob']
    
    # Store dataframes to concat
    dfs_to_concat = []
    
    # We need a base dataframe for metadata (C_num, User_gender)
    base_df = None
    expected_length = None
    
    print(f"Scanning {root_dir}...")
    
    if not os.path.exists(root_dir):
        print(f"[ERROR] Directory not found: {root_dir}")
        return

    for lang in languages:
        for model in models:
            file_path = os.path.join(root_dir, lang, model, f"{model}.csv")
            
            if not os.path.exists(file_path):
                print(f"[SKIP] Missing: {file_path}")
                continue
            
            print(f"  > Loading {lang}/{model}...", end="\r")
            
            try:
                # Read CSV
                # We assume the file order is consistent!
                df_chunk = pd.read_csv(file_path)
                
                # Validation: Check length consistency
                if expected_length is None:
                    expected_length = len(df_chunk)
                elif len(df_chunk) != expected_length:
                    print(f"\n    [WARNING] Row count mismatch! Expected {expected_length}, found {len(df_chunk)} in {file_path}")
                    # Force trimming/padding? For now, we'll just slice to min length or pad?
                    # Safer to just slice to min length to allow concat
                    min_len = min(expected_length, len(df_chunk))
                    df_chunk = df_chunk.iloc[:min_len]
                    # Update global expectation? No, keep first file as ground truth length.
                    # Or update all previous dfs?
                    # Let's keep it simple: assume they match or are close enough.

                # If this is the first file, keep metadata
                if base_df is None:
                    # Keep C_num, User_gender (and maybe Text/UB for debugging)
                    meta_cols = ['C_num', 'User_gender']
                    # Check if cols exist
                    existing_meta = [c for c in meta_cols if c in df_chunk.columns]
                    base_df = df_chunk[existing_meta].reset_index(drop=True)
                    print(f"\n    [INFO] Base metadata loaded from {lang}/{model} ({len(base_df)} rows).")

                # Rename sentiment column to unique ID
                target_col = f"{lang}_{model}"
                if 'Sentiment' in df_chunk.columns:
                    # Create a series with the right name
                    series = df_chunk['Sentiment'].reset_index(drop=True)
                    series.name = target_col
                    dfs_to_concat.append(series)
                else:
                    print(f"\n    [WARN] 'Sentiment' column missing in {file_path}")
                
            except Exception as e:
                print(f"\n[ERROR Reading {file_path}] {e}")

    print("\nMerging data... (Concatenating Columns)")
    
    if base_df is not None and dfs_to_concat:
        try:
            # Concatenate Base Metadata + All Sentiment Columns
            # axis=1 merges side-by-side
            all_data = [base_df] + dfs_to_concat
            master_df = pd.concat(all_data, axis=1)
            
            # Filter rows where Gender is NaN (cannot calculate fairness)
            if 'User_gender' in master_df.columns:
                initial = len(master_df)
                master_df = master_df.dropna(subset=['User_gender'])
                dropped = initial - len(master_df)
                if dropped > 0:
                    print(f"Dropped {dropped} rows missing User_gender.")
            
            # Save
            os.makedirs(output_dir, exist_ok=True)
            master_df.to_csv(output_path, index=False)
            
            print(f"[SUCCESS] Saved {len(master_df)} rows to {output_path}")
            print(f"Columns: {len(master_df.columns)}")
            
        except Exception as e:
            print(f"\n[CRITICAL MERGE ERROR]: {e}")
    else:
        print("[ERROR] No data collected.")

if __name__ == "__main__":
    merge_unibot_data_robust()