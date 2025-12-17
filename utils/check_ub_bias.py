import pandas as pd
from scipy.stats import chi2_contingency
import os

def check_ub_gender_correlation():
    # Path to your file
    data_path = "data/input/real_world/master_sentiment_unibot.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check if columns exist
    if 'User_gender' not in df.columns or 'C_num' not in df.columns:
        print("Error: Columns 'User_gender' or 'C_num' missing.")
        # It seems your CSV might have 'C_num' instead of 'UB' based on snippets?
        # Let's check based on your description that UB distinguishes User vs Bot.
        # If UB is missing, we try to infer it or print available columns.
        print(f"Available columns: {df.columns.tolist()}")
        return

    # 1. Create Contingency Table
    # Rows: User_gender, Cols: C_num
    contingency = pd.crosstab(df['User_gender'], df['C_num'])
    
    print("\n=== Contingency Table (Counts) ===")
    print(contingency)
    
    # 2. Chi-Square Test of Independence
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    print(f"\n=== Chi-Square Test Results ===")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-Value:        {p:.4e}")
    
    if p < 0.05:
        print("\n>>> CONCLUSION: SIGNIFICANT CORRELATION DETECTED.")
        print("    User_gender affects C_num (Text Type).")
        print("    This confirms Z -> T exists in your dataset.")
        print("    Your strategy to filter for 'Bot Only' is scientifically necessary.")
    else:
        print("\n>>> CONCLUSION: NO SIGNIFICANT CORRELATION.")
        print("    User_gender and C_num appear independent.")
    
    # 3. Normalized Proportions (for easier reading)
    print("\n=== Proportions (Row-wise %) ===")
    print(pd.crosstab(df['User_gender'], df['C_num'], normalize='index') * 100)

if __name__ == "__main__":
    check_ub_gender_correlation()