import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from io import StringIO

# --- Data Processing and Plotting Function ---
def create_histograms(df, bins=20):
    """
    Filters the DataFrame, calculates log returns, and generates two histograms.

    Args:
        df (pd.DataFrame): The input DataFrame loaded from the CSV.
        bins (int): The number of bins to use for the histogram.
    """
    # Ensure 'profitable' is boolean and convert 'blackScholesPrice' to numeric
    df['profitable'] = df['profitable'].astype(bool)
    df['blackScholesPrice'] = pd.to_numeric(df['blackScholesPrice'], errors='coerce')

    # --- 1. Filter the Data ---

    # Filter 1: All profitable rows (must be True)
    profitable_df = df[df['profitable'] == True].dropna(subset=['return'])
    
    # Filter 2: Profitable rows AND Black-Scholes price > Midpoint
    profitable_and_bs_gt_mid = profitable_df[
        profitable_df['blackScholesPrice'] > profitable_df['midpoint']
    ].dropna(subset=['return'])


    # --- 2. Calculate Log Returns ---
    # Filter out returns < -1 before taking the log
    profitable_df = profitable_df[profitable_df['return'] > -1]
    profitable_and_bs_gt_mid = profitable_and_bs_gt_mid[profitable_and_bs_gt_mid['return'] > -1]

    log_returns_all_profitable = np.log1p(profitable_df['return'])
    log_returns_bs_gt_mid = np.log1p(profitable_and_bs_gt_mid['return'])
        
    # --- 3. Generate and Display Histograms ---

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: All Profitable Returns Histogram ---
    if not log_returns_all_profitable.empty:
        # Use plt.hist() to generate the histogram
        axes[0].hist(log_returns_all_profitable, bins=bins, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'Histogram: Log Returns (All Profitable Rows)\n(n={len(log_returns_all_profitable)})')
        axes[0].set_xlabel('Log Return ($\ln(1 + \text{Return})$)')
        axes[0].set_ylabel('Frequency (Count)')
    else:
        axes[0].set_title('Histogram 1: Insufficient Data')

    # --- Plot 2: Profitable & BS > Midpoint Histogram ---
    if not log_returns_bs_gt_mid.empty:
        # Use plt.hist() to generate the histogram
        axes[1].hist(log_returns_bs_gt_mid, bins=bins, edgecolor='black', alpha=0.7, color='C1')
        axes[1].set_title(f'Histogram: Log Returns (Profitable & BS > Midpoint)\n(n={len(log_returns_bs_gt_mid)})')
        axes[1].set_xlabel('Log Return ($\ln(1 + \text{Return})$)')
        axes[1].set_ylabel('Frequency (Count)')
    else:
        axes[1].set_title('Histogram 2: Insufficient Data')
        
    plt.tight_layout()
    plt.show()

# ====================================================================

def main():
    """
    Parses command-line arguments and executes the histogram generation.
    """
    parser = argparse.ArgumentParser(
        description="Generates histograms for log returns from a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define the CSV file argument
    parser.add_argument(
        'csv_file', 
        type=str, 
        help="Path to the input CSV file containing options data."
    )
    
    args = parser.parse_args()
    csv_file_path = args.csv_file

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Ensure the essential columns exist
        required_cols = ['profitable', 'return', 'blackScholesPrice', 'midpoint']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV file must contain the following columns: {required_cols}")
            sys.exit(1)
            
        # Call the plotting function
        create_histograms(df)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':

    # Normal execution starts here after argument handling
    main()