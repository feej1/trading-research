import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from io import StringIO

def analyze_and_graph_groups(df):
    """
    Filters data, creates groups based on the Midpoint/BS Price ratio using fixed bins, 
    and graphs Trade Count, Probability of Profit, and Conditional Average Return.
    
    Args:
        df (pd.DataFrame): The input DataFrame loaded from the CSV.
    """
    # --- 1. Filter Base Data and Prepare Columns ---
    
    required_cols = ['profitable', 'return', 'blackScholesPrice', 'midpoint']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain the following columns: {required_cols}")
        return
        
    df['profitable'] = df['profitable'].astype(bool)
    df['blackScholesPrice'] = pd.to_numeric(df['blackScholesPrice'], errors='coerce')
    df['midpoint'] = pd.to_numeric(df['midpoint'], errors='coerce')
    df['return'] = pd.to_numeric(df['return'], errors='coerce')
    
    # Filter: Only trades where Black Scholes Price is greater than the Midpoint
    filtered_df = df[df['blackScholesPrice'] > df['midpoint']].copy()
    
    if filtered_df.empty:
        print("Error: After filtering (BS Price > Midpoint), no data remains for analysis.")
        return

    # Calculate the ratio: Midpoint / Black Scholes Price (guaranteed to be < 1)
    filtered_df['ratio'] = filtered_df['midpoint'] / filtered_df['blackScholesPrice']
    
    # --- 2. Define Fixed-Width Bins and Labels ---
    
    # Bins: [0.0, 0.80, 0.85, 0.90, 0.95, 1.00]
    bins = [0.0] + [round(i * 0.05, 2) for i in range(16, 21)]
    
    # Labels for the bins
    labels = [
        "< 0.80",
        "0.80 - 0.85",
        "0.85 - 0.90",
        "0.90 - 0.95",
        "0.95 - 1.00"
    ]

    # Create ratio groups using fixed-width cut
    filtered_df['ratio_group'] = pd.cut(
        filtered_df['ratio'], 
        bins=bins, 
        labels=labels, 
        right=True,
        include_lowest=True
    )
    
    # --- 3. Aggregate Metrics per Group ---
    
    # Define a custom lambda function for conditional mean return calculation
    def conditional_mean_return(series):
        # x is the 'return' series for the current group
        # The index is used to look up the 'profitable' status in the main filtered_df
        is_profitable = filtered_df.loc[series.index, 'profitable'] == True
        # Calculate mean return only for trades that were profitable
        return series[is_profitable].mean()

    # Aggregate data by the ratio groups
    grouped_data = filtered_df.groupby('ratio_group', observed=True).agg(
        trade_count=('ratio', 'count'),          
        profitable_count=('profitable', 'sum'),
        avg_return_profitable=('return', conditional_mean_return) 
    ).reset_index()

    # Calculate Probability of Profit (Total Profitable / Total Trades)
    grouped_data['prob_of_profit'] = grouped_data['profitable_count'] / grouped_data['trade_count']
    grouped_data['prob_of_profit'] = grouped_data['prob_of_profit'].fillna(0.0)
    
    # Handle NaN returns (groups with profitable_count = 0)
    grouped_data['avg_return_profitable'] = grouped_data['avg_return_profitable'].fillna(0.0)

    # --- 4. Graph Results (Three Separate Figures) ---
    
    x_labels = grouped_data['ratio_group']
    
    TITLE_FONT_SIZE = 14
    LABEL_FONT_SIZE = 12
    ANNOTATION_FONT_SIZE = 10
    FIGURE_SIZE = (10, 5) # Set ideal size for a single, focused plot
    
    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Plot 1: Trade Count (Volume) ---
    fig1, ax1 = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    bars_count = ax1.bar(x_labels, grouped_data['trade_count'], color='tab:orange', alpha=0.7)
    
    ax1.set_title('1. Trade Count per Ratio Group (BS Price > Midpoint)', fontsize=TITLE_FONT_SIZE)
    ax1.set_ylabel('Number of Trades', fontsize=LABEL_FONT_SIZE)
    ax1.set_xlabel('Midpoint / Black Scholes Price Ratio Group', fontsize=LABEL_FONT_SIZE)
    
    for bar in bars_count:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=ANNOTATION_FONT_SIZE)
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout() 

    # --- Plot 2: Probability of Profit ---
    fig2, ax2 = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    bars_prob = ax2.bar(x_labels, grouped_data['prob_of_profit'], color='tab:green', alpha=0.7)
    
    ax2.set_title('2. Probability of Profit per Ratio Group', fontsize=TITLE_FONT_SIZE)
    ax2.set_ylabel('Probability (Ratio of Profitable/Total)', fontsize=LABEL_FONT_SIZE)
    ax2.set_xlabel('Midpoint / Black Scholes Price Ratio Group', fontsize=LABEL_FONT_SIZE)
    ax2.set_ylim(0, 1) # Probability axis from 0 to 1
    
    for bar in bars_prob:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=ANNOTATION_FONT_SIZE)
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout() 


    # --- Plot 3: Average Return When Profitable ---
    fig3, ax3 = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    bars_return = ax3.bar(x_labels, grouped_data['avg_return_profitable'], color='tab:blue', alpha=0.7)
    
    ax3.set_title('3. Average Return (Profit) When Profitable', fontsize=TITLE_FONT_SIZE)
    ax3.set_ylabel('Average Return (%)', fontsize=LABEL_FONT_SIZE)
    ax3.set_xlabel('Midpoint / Black Scholes Price Ratio Group', fontsize=LABEL_FONT_SIZE)
    
    for bar in bars_return:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=ANNOTATION_FONT_SIZE)
    
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout() 
    
    # Show all three figures
    plt.show()

# ====================================================================

def main():
    """Parses command-line arguments and executes the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyzes trades filtered by BS Price > Midpoint, grouped by fixed-width bins of the Midpoint/BS Price ratio.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('csv_file', type=str, help="Path to the input CSV file containing options data.")
    args = parser.parse_args()
    csv_file_path = args.csv_file

    try:
        df = pd.read_csv(csv_file_path)
        analyze_and_graph_groups(df)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        # This handles the case where the dummy file is processed for demonstration
        if not csv_file_path.startswith("dummy"):
             sys.exit(1)


if __name__ == '__main__':
    main()
