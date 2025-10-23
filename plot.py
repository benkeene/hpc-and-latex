"""
Generate 3D surface plot of final test loss from experiment results.

Usage:
    python plot.py ./results/simulation_01/results.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_surface(csv_path):
    """
    Create a 3D surface plot of final test loss.

    Args:
        csv_path: Path to the results CSV file

    The plot shows:
        - X-axis: widths
        - Y-axis: n_samples
        - Z-axis: final_test_loss (averaged over repetitions)
    """
    # Load the results
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} experiments from {csv_path}")

    # Group by widths and n_samples, average over repetitions
    grouped = df.groupby(['widths', 'n_samples'])['final_test_loss'].mean().reset_index()

    print(f"Averaged over repetitions, {len(grouped)} unique (width, n_samples) combinations")

    # Count number of unique repetitions for the title
    num_repetitions = df['repetitions'].nunique()

    # Pivot to create a 2D grid for surface plot (swap widths and n_samples)
    pivot = grouped.pivot(index='widths', columns='n_samples', values='final_test_loss')

    # Get the unique values for x and y axes (swapped)
    n_samples = pivot.columns.values
    widths = pivot.index.values

    # Create meshgrid with natural logarithmic values (X is now n_samples, Y is widths)
    X, Y = np.meshgrid(np.log(n_samples), np.log(widths))
    Z = np.log(pivot.values)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

    # Add labels and title
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Width', fontsize=12)
    ax.set_zlabel('Final Test Loss', fontsize=12, labelpad=15)
    ax.set_title(f'Final Test Loss vs Width and Number of Samples (Log-Log-Log)\n(Averaged over {num_repetitions} Repetitions)',
                 fontsize=14, pad=20)

    # Set tick marks to only show actual data values (in log scale)
    ax.set_xticks(np.log(n_samples))
    ax.set_xticklabels([f'{s:.0f}' for s in n_samples])
    ax.set_yticks(np.log(widths))
    ax.set_yticklabels([f'{w:.0f}' for w in widths])

    # Set z-axis ticks to show actual loss values (not log values)
    # Get the range of log(loss) values to determine tick positions
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    # Create evenly spaced ticks in log space
    z_tick_positions = np.linspace(z_min, z_max, 5)
    # Convert back to actual loss values for labels
    z_tick_labels = [f'{np.exp(z):.6f}' for z in z_tick_positions]
    ax.set_zticks(z_tick_positions)
    ax.set_zticklabels(z_tick_labels)

    # Add padding to z-axis tick labels to prevent collision
    ax.tick_params(axis='z', pad=8)

    # Adjust viewing angle for better visibility
    ax.view_init(elev=25, azim=45)

    # Save the plot to ./plots/{expname}/testloss.pdf
    csv_parent = Path(csv_path).parent
    expname = csv_parent.name
    plots_dir = Path('plots') / expname
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_path = plots_dir / 'testloss.pdf'

    # Use subplots_adjust instead of tight_layout for better control with 3D plots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"\nPlot saved to: {output_path}")

    # Display the plot
    plt.show()


def plot_training_history(results_dir):
    """
    Create training history plots for experiments in the history directory.

    Args:
        results_dir: Path to the results directory (e.g., './results/simulation_01')

    The function looks for CSV files in results_dir/history/ and creates a plot
    for each one, showing train and test loss over epochs.
    """
    results_path = Path(results_dir)
    history_dir = results_path / 'history'

    if not history_dir.exists():
        print(f"No history directory found at {history_dir}")
        return

    # Find all CSV files in the history directory
    csv_files = sorted(history_dir.glob('exp_*_history.csv'))

    if not csv_files:
        print(f"No history CSV files found in {history_dir}")
        return

    print(f"Found {len(csv_files)} history files in {history_dir}")

    # Create output directory for plots
    expname = results_path.name
    plots_dir = Path('plots') / expname
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Process each history file
    for csv_file in csv_files:
        # Extract experiment index from filename (e.g., exp_0_history.csv -> 0)
        exp_idx = csv_file.stem.split('_')[1]

        # Load the history data
        df = pd.read_csv(csv_file)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot train and test loss
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2, alpha=0.8)
        ax.plot(df['epoch'], df['test_loss'], label='Test Loss', linewidth=2, alpha=0.8)

        # Add labels and title
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training History - Experiment {exp_idx}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Use log scale for y-axis if losses span multiple orders of magnitude
        loss_range = df[['train_loss', 'test_loss']].max().max() / df[['train_loss', 'test_loss']].min().min()
        if loss_range > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Loss (log scale)', fontsize=12)

        # Save the plot
        output_path = plots_dir / f'history_{exp_idx}.pdf'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot to: {output_path}")


def main():
    """Main entry point for the plot script."""
    if len(sys.argv) != 2:
        print("Usage: python plot.py <path_to_results_dir>")
        print("Example: python plot.py ./results/simulation_01")
        sys.exit(1)

    results_dir = sys.argv[1]
    path = Path(results_dir)

    # Check if directory exists
    if not path.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    if not path.is_dir():
        print(f"Error: Input must be a directory, not a file")
        print(f"Provided: {results_dir}")
        sys.exit(1)

    # Generate training history plots
    print("Generating training history plots...")
    plot_training_history(results_dir)

    # Generate surface plot from results.csv
    results_csv = path / 'results.csv'
    if results_csv.exists():
        print("\nGenerating surface plot...")
        plot_surface(str(results_csv))
    else:
        print(f"\nWarning: results.csv not found at {results_csv}")
        print("Skipping surface plot generation.")


if __name__ == "__main__":
    main()
