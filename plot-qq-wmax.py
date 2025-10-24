import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from io import StringIO
# Using scipy.stats for the Generalized Extreme Value distribution (GEV)
from scipy.stats import genextreme 
from statsmodels.graphics.gofplots import qqplot_2samples 

# --- Data Processing and Plotting Function ---
def create_gev_qq_plots(df):
    """
    Filters the DataFrame, calculates log returns, estimates GEV parameters 
    using scipy.stats.genextreme.fit(), and generates two Q-Q plots.

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
        
    # --- 2. Generate and Display Q-Q Plots against GEV Distribution ---

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Helper function for fitting parameters using scipy.stats ---
    def fit_gev_params_scipy(data):
        # genextreme.fit performs MLE and returns (c, loc, scale)
        # c is the shape parameter (xi), loc is location (mu), scale is scale (beta)
        params = genextreme.fit(data.values)
        return params[0], params[1], params[2] # c (xi), loc (mu), scale (beta)

    # --- Plot 1: All Profitable Returns ---
    if not log_returns_all_profitable.empty:
        # 2a. Estimate GEV parameters (c, loc, scale)
        c_all, loc_all, scale_all = fit_gev_params_scipy(log_returns_all_profitable)

        # 2b. Generate Theoretical GEV Quantiles for the plot
        # genextreme.rvs generates random variables from the fitted GEV distribution
        theoretical_gev = genextreme.rvs(c=c_all, loc=loc_all, scale=scale_all, size=len(log_returns_all_profitable))
        
        # 2c. Plotting the Q-Q plot 
        qqplot_2samples(
            log_returns_all_profitable, 
            theoretical_gev, 
            line='45', 
            ax=axes[0]
        )
        # Use xi for the shape parameter in the title
        axes[0].set_title(f"Q-Q Plot: Log Returns vs GEV\n($\\xi={c_all:.4f}$ | All Profitable Rows)")
        axes[0].set_xlabel("Theoretical Quantiles (GEV)")
        axes[0].set_ylabel('Sample Log Returns')
    else:
        axes[0].set_title('Q-Q Plot 1: Insufficient Data')

    # --- Plot 2: Profitable & BS > Midpoint ---
    if not log_returns_bs_gt_mid.empty:
        # 2a. Estimate GEV parameters (c, loc, scale)
        c_bs, loc_bs, scale_bs = fit_gev_params_scipy(log_returns_bs_gt_mid)

        # 2b. Generate Theoretical GEV Quantiles for the plot
        theoretical_gev = genextreme.rvs(c=c_bs, loc=loc_bs, scale=scale_bs, size=len(log_returns_bs_gt_mid))

        # 2c. Plotting the Q-Q plot 
        qqplot_2samples(
            log_returns_bs_gt_mid, 
            theoretical_gev, 
            line='45', 
            ax=axes[1]
        )
        axes[1].set_title(f"Q-Q Plot: Log Returns vs GEV\n($\\xi={c_bs:.4f}$ | Profitable & BS > Midpoint)")
        axes[1].set_xlabel("Theoretical Quantiles (GEV)")
        axes[1].set_ylabel('Sample Log Returns')
    else:
        axes[1].set_title('Q-Q Plot 2: Insufficient Data')
        
    plt.tight_layout()
    plt.show()

# ====================================================================

def main():
    """Parses command-line arguments and executes the Q-Q plot generation."""
    parser = argparse.ArgumentParser(
        description="Generates Generalized Extreme Value (GEV) Q-Q plots for log returns from a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('csv_file', type=str, help="Path to the input CSV file containing options data.")
    args = parser.parse_args()
    csv_file_path = args.csv_file

    try:
        df = pd.read_csv(csv_file_path)
        required_cols = ['profitable', 'return', 'blackScholesPrice', 'midpoint']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV file must contain the following columns: {required_cols}")
            sys.exit(1)
            
        create_gev_qq_plots(df)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        # Print the exception details only if it's not the dummy block logic
        if not csv_file_path.startswith("dummy"):
             sys.exit(1)


if __name__ == '__main__':
    main()