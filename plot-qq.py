#!/usr/bin/env python3
"""Create a QQ-plot of returns for profitable options.

Reads a CSV (default: SPY_2018-01-01_2018-06-01_values.csv), filters out rows
where `profitable` is false, then creates and saves a QQ-plot of the
`return` column (assumed to be percentage values).

Usage:
    python plot-qq.py --csv path/to/file.csv --out output.png
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser(description="QQ plot of returns for profitable options")
    p.add_argument('--csv', '-c', type=str, default='SPY_2018-01-01_2018-06-01_values.csv', help='CSV file to load')
    p.add_argument('--out', '-o', type=str, default=None, help='Output image path (PNG). Defaults to <csv_basename>_qq.png')
    p.add_argument('--show', action='store_true', help='Show the plot interactively')
    return p.parse_args()


def profitable_mask(val):
    # Accept boolean, numeric (1/0), and string representations
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    if s in ('true', 't', 'yes', 'y', '1'):
        return True
    try:
        return float(s) != 0.0
    except Exception:
        return False


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        sys.exit(2)

    df = pd.read_csv(args.csv)

    if 'profitable' not in df.columns:
        print("CSV does not contain 'profitable' column")
        sys.exit(2)

    mask = df['profitable'].apply(profitable_mask)
    df = df[mask]

    if df.empty:
        print('No profitable rows found after filtering; nothing to plot.')
        sys.exit(0)

    if 'return' not in df.columns:
        print("CSV does not contain 'return' column")
        sys.exit(2)

    # Coerce return values to numeric and drop NaNs
    returns = pd.to_numeric(df['return'], errors='coerce').dropna()

    if returns.empty:
        print('No numeric return values to plot.')
        sys.exit(0)

    # Compute ordered sample and theoretical quantiles using scipy.stats.probplot
    (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist='norm')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(osm, osr, s=10, alpha=0.6)
    ax.plot(osm, slope * osm + intercept, 'r--', label=f'fit: y={slope:.3f}x+{intercept:.3f}')
    ax.set_xlabel('Theoretical quantiles (normal)')
    ax.set_ylabel('Sample quantiles (returns)')
    ax.set_title('QQ-plot of returns (profitable only)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    out_path = args.out
    if not out_path:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = f"{base}_qq.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'QQ-plot saved to: {out_path}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()


