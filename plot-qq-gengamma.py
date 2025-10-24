import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from io import StringIO
# Using scipy.stats for the Generalized Gamma distribution (gengamma)
from scipy.stats import gengamma 
from statsmodels.graphics.gofplots import qqplot_2samples 

# --- Data Processing and Plotting Function ---
def create_gengamma_qq_plots(df):
    """
    Filters the DataFrame, calculates log returns, estimates Generalized Gamma parameters, 
    and generates two Q-Q plots against the estimated distribution.

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
        
    # --- 2. Generate and Display Q-Q Plots against Generalized Gamma Distribution ---

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # --- Plot 1: All Profitable Returns ---
    if not log_returns_all_profitable.empty:
        data_shift = log_returns_all_profitable.values - log_returns_all_profitable.min() + 1e-6
        params_all_prof = gengamma.fit(data_shift)
        a_all_prof, c_all_prof = params_all_prof[0], params_all_prof[1]
        theoretical_gengamma = gengamma.rvs(*params_all_prof, size=len(data_shift))
        qqplot_2samples(
            data_shift, 
            theoretical_gengamma, 
            line='45', 
            ax=axes[0]
        )
        axes[0].set_title(f'Q-Q Plot: Log Returns vs Generalized Gamma\n($a={a_all_prof:.2f}, c={c_all_prof:.2f}$ | All Profitable)')
        axes[0].set_xlabel('Theoretical Quantiles (Gen. Gamma)')
        axes[0].set_ylabel('Sample Log Returns (Shifted)')
    else:
        axes[0].set_title('Q-Q Plot 1: Insufficient Data')

    # --- Plot 2: Profitable & BS > Midpoint ---
    if not log_returns_bs_gt_mid.empty:
        data_shift = log_returns_bs_gt_mid.values - log_returns_bs_gt_mid.min() + 1e-6
        params_bs_gt_mid = gengamma.fit(data_shift)
        a_bs_gt_mid, c_bs_gt_mid = params_bs_gt_mid[0], params_bs_gt_mid[1]
        theoretical_gengamma = gengamma.rvs(*params_bs_gt_mid, size=len(data_shift))
        qqplot_2samples(
            data_shift, 
            theoretical_gengamma, 
            line='45', 
            ax=axes[1]
        )
        axes[1].set_title(f'Q-Q Plot: Log Returns vs Generalized Gamma\n($a={a_bs_gt_mid:.2f}, c={c_bs_gt_mid:.2f}$ | Profitable & BS > Mid)')
        axes[1].set_xlabel('Theoretical Quantiles (Gen. Gamma)')
        axes[1].set_ylabel('Sample Log Returns (Shifted)')
    else:
        axes[1].set_title('Q-Q Plot 2: Insufficient Data')

    # --- Plot 3: Profitable & BS > Midpoint & Top Quartile Ratio ---
    # Compute ratio and filter for top quartile
    ratio = profitable_and_bs_gt_mid['midpoint'] / profitable_and_bs_gt_mid['blackScholesPrice']
    q3 = ratio.quantile(0.75)
    mask_top_quartile = ratio >= q3
    top_quartile_df = profitable_and_bs_gt_mid[mask_top_quartile]
    log_returns_top_quartile = np.log1p(top_quartile_df['return'])
    if not log_returns_top_quartile.empty:
        data_shift = log_returns_top_quartile.values - log_returns_top_quartile.min() + 1e-6
        params_top_quartile = gengamma.fit(data_shift)
        a_top_quartile, c_top_quartile = params_top_quartile[0], params_top_quartile[1]
        theoretical_gengamma = gengamma.rvs(*params_top_quartile, size=len(data_shift))
        qqplot_2samples(
            data_shift, 
            theoretical_gengamma, 
            line='45', 
            ax=axes[2]
        )
        axes[2].set_title(f'Q-Q Plot: Log Returns vs Gen. Gamma\n($a={a_top_quartile:.2f}, c={c_top_quartile:.2f}$ | Top Quartile Ratio)')
        axes[2].set_xlabel('Theoretical Quantiles (Gen. Gamma)')
        axes[2].set_ylabel('Sample Log Returns (Shifted)')
    else:
        axes[2].set_title('Q-Q Plot 3: Insufficient Data')

    plt.tight_layout()
    plt.show()

# ====================================================================

def main():
    """Parses command-line arguments and executes the Q-Q plot generation."""
    parser = argparse.ArgumentParser(
        description="Generates Generalized Gamma Q-Q plots for log returns from a CSV file.",
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
            
        create_gengamma_qq_plots(df)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()