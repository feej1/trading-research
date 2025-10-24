import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from io import StringIO
# Re-introduce scipy.stats, specifically for fitting the Skew Normal distribution.
from scipy.stats import skewnorm
from statsmodels.graphics.gofplots import qqplot_2samples 

# --- Data Processing and Plotting Function ---
def create_skewnorm_qq_plots(df):
    """
    Filters the DataFrame, calculates log returns, estimates Skew Normal parameters, 
    and generates two Q-Q plots against the estimated Skew Normal distribution.

    Args:
        df (pd.DataFrame): The input DataFrame loaded from the CSV.
    """
    # Ensure data types are correct
    df['profitable'] = df['profitable'].astype(bool)
    df['blackScholesPrice'] = pd.to_numeric(df['blackScholesPrice'], errors='coerce')

    # --- 1. Filter and Prepare Data ---
    profitable_df = df[df['profitable'] == True].dropna(subset=['return'])
    profitable_and_bs_gt_mid = profitable_df[
        profitable_df['blackScholesPrice'] > profitable_df['midpoint']
    ].dropna(subset=['return'])

    # Filter out returns <= -1 for log calculation
    profitable_df = profitable_df[profitable_df['return'] > -1]
    profitable_and_bs_gt_mid = profitable_and_bs_gt_mid[profitable_and_bs_gt_mid['return'] > -1]

    log_returns_all_profitable = np.log1p(profitable_df['return'])
    log_returns_bs_gt_mid = np.log1p(profitable_and_bs_gt_mid['return'])
        
    # --- 2. Generate and Display Q-Q Plots against Skew Normal Distribution ---

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: All Profitable Returns ---
    if not log_returns_all_profitable.empty:
        # 2a. Estimate Skew Normal parameters (a, loc, scale)
        params_all_prof = skewnorm.fit(log_returns_all_profitable.values)
        a_all_prof = params_all_prof[0] # 'a' is the skewness parameter

        # 2b. Generate Theoretical Skew Normal Quantiles for the plot
        # We sample points from the theoretical Skew Normal distribution
        theoretical_skewnorm = skewnorm.rvs(*params_all_prof, size=len(log_returns_all_profitable))
        
        # 2c. Plotting the Q-Q plot against the Skew Normal quantiles
        qqplot_2samples(
            log_returns_all_profitable, 
            theoretical_skewnorm, 
            line='45', 
            ax=axes[0]
        )
        axes[0].set_title(f'Q-Q Plot: Log Returns vs Skew Normal ($\mathbf{{a={a_all_prof:.2f}}}$)\n(All Profitable Rows)')
        axes[0].set_xlabel('Theoretical Quantiles (Skew Normal)')
        axes[0].set_ylabel('Sample Log Returns')
    else:
        axes[0].set_title('Q-Q Plot 1: Insufficient Data')

    # --- Plot 2: Profitable & BS > Midpoint ---
    if not log_returns_bs_gt_mid.empty:
        # 2a. Estimate Skew Normal parameters (a, loc, scale)
        params_bs_gt_mid = skewnorm.fit(log_returns_bs_gt_mid.values)
        a_bs_gt_mid = params_bs_gt_mid[0]

        # 2b. Generate Theoretical Skew Normal Quantiles for the plot
        theoretical_skewnorm = skewnorm.rvs(*params_bs_gt_mid, size=len(log_returns_bs_gt_mid))

        # 2c. Plotting the Q-Q plot against the Skew Normal quantiles
        qqplot_2samples(
            log_returns_bs_gt_mid, 
            theoretical_skewnorm, 
            line='45', 
            ax=axes[1]
        )
        axes[1].set_title(f'Q-Q Plot: Log Returns vs Skew Normal ($\mathbf{{a={a_bs_gt_mid:.2f}}}$)\n(Profitable & BS > Midpoint)')
        axes[1].set_xlabel('Theoretical Quantiles (Skew Normal)')
        axes[1].set_ylabel('Sample Log Returns')
    else:
        axes[1].set_title('Q-Q Plot 2: Insufficient Data')
        
    plt.tight_layout()
    plt.show()

# ====================================================================

def main():
    """Parses command-line arguments and executes the Q-Q plot generation."""
    parser = argparse.ArgumentParser(
        description="Generates Skew Normal Q-Q plots for log returns from a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'csv_file', 
        type=str, 
        help="Path to the input CSV file containing options data."
    )
    
    args = parser.parse_args()
    csv_file_path = args.csv_file

    try:
        df = pd.read_csv(csv_file_path)
        required_cols = ['profitable', 'return', 'blackScholesPrice', 'midpoint']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV file must contain the following columns: {required_cols}")
            sys.exit(1)
            
        create_skewnorm_qq_plots(df)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':

    main()