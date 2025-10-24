import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from io import StringIO
# Using scipy.stats for the Log Normal distribution (lognorm)
from scipy.stats import lognorm 
from statsmodels.graphics.gofplots import qqplot_2samples 

# --- Data Processing and Plotting Function ---
def create_lognorm_qq_plots(df):
    """
    Filters the DataFrame, calculates log returns, estimates Log Normal parameters 
    using scipy.stats.lognorm.fit(), and generates two Q-Q plots.

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
        
    # --- 2. Generate and Display Q-Q Plots against Log Normal Distribution ---

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Helper function for fitting parameters using scipy.stats ---
    def fit_lognorm_params_scipy(data):
        # lognorm requires input data to be strictly positive (> 0)
        # Log returns can be negative, so we must shift them for fitting.
        data_shift = data.values - data.min() + 1e-6
        # lognorm.fit performs MLE and returns (s, loc, scale)
        params = lognorm.fit(data_shift)
        return params[0], params[1], params[2] # s, loc, scale

    # --- Plot 1: All Profitable Returns ---
    if not log_returns_all_profitable.empty:
        # Shift data for fitting (Log Normal requires positive data)
        data_shift = log_returns_all_profitable.values - log_returns_all_profitable.min() + 1e-6
        
        # 2a. Estimate Log Normal parameters (s, loc, scale)
        s_all, loc_all, scale_all = fit_lognorm_params_scipy(log_returns_all_profitable)

        # 2b. Generate Theoretical Log Normal Quantiles for the plot
        # lognorm.rvs generates random variables from the fitted Log Normal distribution
        theoretical_lognorm = lognorm.rvs(s=s_all, loc=loc_all, scale=scale_all, size=len(data_shift))
        
        # 2c. Plotting the Q-Q plot (using the shifted data)
        qqplot_2samples(
            data_shift, 
            theoretical_lognorm, 
            line='45', 
            ax=axes[0]
        )
        axes[0].set_title(f"Q-Q Plot: Log Returns vs Log Normal\n($s={s_all:.2f}$ | All Profitable Rows)")
        axes[0].set_xlabel("Theoretical Quantiles (Log Normal)")
        axes[0].set_ylabel('Sample Log Returns (Shifted)')
    else:
        axes[0].set_title('Q-Q Plot 1: Insufficient Data')

    # --- Plot 2: Profitable & BS > Midpoint ---
    if not log_returns_bs_gt_mid.empty:
        # Shift data for fitting
        data_shift = log_returns_bs_gt_mid.values - log_returns_bs_gt_mid.min() + 1e-6
        
        # 2a. Estimate Log Normal parameters (s, loc, scale)
        s_bs, loc_bs, scale_bs = fit_lognorm_params_scipy(log_returns_bs_gt_mid)

        # 2b. Generate Theoretical Log Normal Quantiles for the plot
        theoretical_lognorm = lognorm.rvs(s=s_bs, loc=loc_bs, scale=scale_bs, size=len(data_shift))

        # 2c. Plotting the Q-Q plot 
        qqplot_2samples(
            data_shift, 
            theoretical_lognorm, 
            line='45', 
            ax=axes[1]
        )
        axes[1].set_title(f"Q-Q Plot: Log Returns vs Log Normal\n($s={s_bs:.2f}$ | Profitable & BS > Midpoint)")
        axes[1].set_xlabel("Theoretical Quantiles (Log Normal)")
        axes[1].set_ylabel('Sample Log Returns (Shifted)')
    else:
        axes[1].set_title('Q-Q Plot 2: Insufficient Data')
        
    plt.tight_layout()
    plt.show()

# ====================================================================

def main():
    """Parses command-line arguments and executes the Q-Q plot generation."""
    parser = argparse.ArgumentParser(
        description="Generates Log Normal Q-Q plots for log returns from a CSV file.",
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
            
        create_lognorm_qq_plots(df)

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