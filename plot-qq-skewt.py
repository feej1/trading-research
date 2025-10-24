import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from io import StringIO
from scipy.stats import jf_skew_t, probplot

def create_skewed_t_qq_plots(df):
    # --- Data typing and filtering (adapt to your column names) ---
    df = df.copy()
    # Ensure types; adjust column names if different
    if "profitable" in df.columns:
        df["profitable"] = df["profitable"].astype(bool)
    if "return" not in df.columns:
        raise ValueError("DataFrame must contain a 'return' column.")

    # Keep reasonable returns (> -1 for log1p)
    base = df[df["return"] > -1].dropna(subset=["return"])

    # Two groups you previously used
    group_all = base[base.get("profitable", True)]  # if profitable exists, filter it
    if "blackScholesPrice" in base.columns and "midpoint" in base.columns:
        group_bs = group_all[group_all["blackScholesPrice"] > group_all["midpoint"]]
    else:
        group_bs = group_all  # fallback: just use same

    # Compute log returns (use log1p for stability)
    data_all = np.log1p(group_all["return"].values).astype(float)
    data_bs = np.log1p(group_bs["return"].values).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    def fit_and_qq(data, ax, title):
        if len(data) < 10:
            ax.set_title(f"{title} — insufficient data (n={len(data)})")
            return

        # fit returns (a, b, loc, scale)
        try:
            a, b, loc, scale = jf_skew_t.fit(data)
            frozen = jf_skew_t(a, b, loc=loc, scale=scale)  # frozen distribution
            probplot(data, dist=frozen, plot=ax)
            ax.set_title(f"{title}\njf_skew_t fit: a={a:.3f}, b={b:.3f}, loc={loc:.4f}, scale={scale:.4f}")
            ax.set_xlabel("Theoretical Quantiles (jf_skew_t)")
            ax.set_ylabel("Sample Quantiles")
            return
        except Exception as e:
            # If fit fails, fall through to ARCH fallback (if available)
            ax.text(0.5, 0.5, f"jf_skew_t fit failed:\n{e}", ha="center", va="center")
            ax.set_title(title)
            # do not return — try fallback below



    fit_and_qq(data_all, axes[0], "All Profitable (log returns)")
    fit_and_qq(data_bs, axes[1], "Profitable & BS > Midpoint (log returns)")

    plt.tight_layout()
    plt.show()
# ====================================================================

def main():
    """Parses command-line arguments and executes the Q-Q plot generation."""
    # (Argparse and execution block remains the same)
    parser = argparse.ArgumentParser(
        description="Generates Student's T Q-Q plots for log returns from a CSV file.",
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
            
        create_skewed_t_qq_plots(df)

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