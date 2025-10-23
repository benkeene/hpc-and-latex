"""
Collate results from multiple experiment YAML files into a single DataFrame.

Usage:
    python collate.py ./results/simulation_01
"""

import sys
import yaml
import pandas as pd
from pathlib import Path


def load_results(results_dir):
    """
    Load all YAML result files from a directory.

    Args:
        results_dir: Path to the results directory

    Returns:
        List of dictionaries, each containing results from one experiment
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Look for YAML files in the runs/ subdirectory
    runs_path = results_path / 'runs'
    if runs_path.exists():
        yaml_files = sorted(runs_path.glob('*.yaml'))
    else:
        # Fallback to looking in the main directory for backwards compatibility
        yaml_files = sorted(results_path.glob('*.yaml'))

    if not yaml_files:
        raise ValueError(f"No YAML files found in {results_dir} or {results_dir}/runs")

    results = []
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            result = yaml.safe_load(f)
            results.append(result)

    return results


def collate_results(results_dir):
    """
    Collate all experiment results into a single DataFrame.

    Args:
        results_dir: Path to the results directory

    Returns:
        pandas DataFrame with all results
    """
    # Load all results
    results = load_results(results_dir)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print(f"Loaded {len(df)} experiments from {results_dir}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDataFrame shape: {df.shape}")

    # Save to results.csv inside the results directory
    results_path = Path(results_dir)
    output_file = results_path / "results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")

    return df


def main():
    """Main entry point for the collate script."""
    if len(sys.argv) != 2:
        print("Usage: python collate.py <path_to_results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Collate results
    df = collate_results(results_dir)

    # Print summary statistics

    # Show basic info about numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print("\nNumeric columns summary:")
        with pd.option_context('display.float_format', '{:.3f}'.format):
            print(df[numeric_cols].describe())


if __name__ == "__main__":
    main()
