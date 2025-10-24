import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from io import StringIO
from fitter import Fitter, get_common_distributions

# --- Data Processing and Fitting Function ---
def find_best_distribution(df):
    """
    Filters the DataFrame, calculates log returns, and finds the best-fitting 
    statistical distribution for the 'All Profitable Rows' dataset.

    Args:
        df (pd.DataFrame): The input DataFrame loaded from the CSV.
    """
    # Ensure data types are correct
    df['profitable'] = df['profitable'].astype(bool)
    
    # --- 1. Filter the Data ---

    # Filter 1: All profitable rows
    profitable_df = df[df['profitable'] == True].dropna(subset=['return'])
    
    # Filter out returns <= -1 for log calculation
    profitable_df = profitable_df[profitable_df['return'] > -1]

    # --- 2. Calculate Log Returns ---
    log_returns = np.log1p(profitable_df['return'])
    
    if log_returns.empty:
        print("Error: After filtering, no sufficient data remains to perform distribution fitting.")
        return

    # --- 3. Automated Distribution Fitting using Fitter ---
    print(f"\n--- Fitting {len(log_returns)} Log Returns (All Profitable Rows) ---")
    
    # Use a curated list of common and relevant distributions 
    # (to speed up the process compared to 'get_common_distributions()')
    distributions_to_test = [
        'norm', 't', 'skewnorm', 'laplace', 'gennorm', 
        'gamma', 'lognorm', 'expon', 'beta', 'logistic'
    ]
    
    # f = Fitter(log_returns.values, distributions=distributions_to_test) 
    f = Fitter(log_returns.values) 
    
    try:
        # Suppress verbose output during fitting
        f.fit(progress=False)
    except Exception as e:
        print(f"Error during distribution fitting: {e}")
        return

    # --- 4. Display Results ---
    
    # Print the summary table, sorted by Sum Square Error (SSE)
    print("\n--- Distribution Fitting Results (Ranked by SSE) ---")
    summary_df = f.summary(Nbest=5)
    print(summary_df)

    # Get the best fit
    best_fit = f.get_best(method='sumsquare_error')
    best_name = list(best_fit.keys())[0]
    best_params = best_fit[best_name]
    
    print(f"\nBest-Fitting Distribution: {best_name.upper()}")
    print(f"   Parameters: {best_params}")

    # --- 5. Optional: Plot the Best Fits ---
    plt.figure(figsize=(10, 6))
    f.hist() # Plot data histogram
    f.plot_pdf(names=[best_name], Nbest=1, lw=2) # Overlay best fit PDF
    plt.title(f'Histogram of Log Returns with Best Fit: {best_name.upper()}')
    plt.xlabel('Log Return')
    plt.ylabel('Density/Frequency')
    plt.show()

# ====================================================================

def main():
    """
    Parses command-line arguments and executes the distribution finding process.
    """
    parser = argparse.ArgumentParser(
        description="Analyzes log returns from a CSV file and finds the best-fitting statistical distribution using the Fitter library.",
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
        required_cols = ['profitable', 'return']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV file must contain the following columns: {required_cols}")
            sys.exit(1)
            
        # Call the fitting function
        find_best_distribution(df)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':

    main()