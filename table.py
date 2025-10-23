"""
Generate LaTeX tables from experiment results.

Usage:
    python table.py ./results/simulation_01/results.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def generate_master_document(expname, output_dir):
    """
    Generate a master LaTeX document that includes all table files.

    Args:
        expname: Name of the experiment
        output_dir: Directory containing the table files
    """
    print("Generating master document...")

    # Find all .tex files in the output directory
    tex_files = sorted(output_dir.glob('*.tex'))

    # Filter out the master file itself if it exists
    tex_files = [f for f in tex_files if f.name != 'tables.tex']

    if not tex_files:
        print("  Warning: No table files found to include")
        return

    # Escape underscores in experiment name for LaTeX
    expname_escaped = expname.replace('_', '\\_')

    # Create master document
    master_content = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}

\\title{{Tables for Experiment: {expname_escaped}}}
\\author{{}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Experiment Tables}}

"""

    # Add input commands for each table file
    for tex_file in tex_files:
        # Use relative path (just the filename)
        master_content += f"\\input{{{tex_file.name}}}\n\n"

    master_content += """\\end{document}
"""

    # Save master document
    output_path = output_dir / "tables.tex"
    with open(output_path, 'w') as f:
        f.write(master_content)

    print(f"  Saved: {output_path}")
    print(f"  Includes {len(tex_files)} table(s)")


def generate_summary_table(df, output_dir):
    """
    Generate a summary statistics table.

    Args:
        df: DataFrame containing results
        output_dir: Directory to save the table
    """
    print("Generating summary statistics table...")

    # Select numeric columns for summary
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    summary = df[numeric_cols].describe()

    # Convert to LaTeX
    latex_table = summary.to_latex(
        float_format="%.4f",
        caption="Summary statistics of all experiments",
        label="tab:summary",
        escape=True
    )

    # Save to file
    output_path = output_dir / "summary_stats.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"  Saved: {output_path}")


def generate_powerlaw_exponents_table(df, output_dir):
    """
    Generate a table showing power-law exponents for test_loss vs width and num_samples.

    Each cell shows two exponents:
    - m_w: exponent for test_loss vs width (calculated between consecutive widths)
    - m_n: exponent for test_loss vs num_samples (calculated between consecutive num_samples)

    Args:
        df: DataFrame containing results with columns: widths, n_samples, final_test_loss
        output_dir: Directory to save the table
    """
    print("Generating power-law exponents table...")

    # Validate required columns
    required_cols = ['widths', 'n_samples', 'final_test_loss']
    if not all(col in df.columns for col in required_cols):
        print(f"  Warning: Required columns {required_cols} not found, skipping...")
        return

    # Get unique sorted values
    widths = sorted(df['widths'].unique())
    num_samples = sorted(df['n_samples'].unique())

    print(f"  Found {len(widths)} unique widths: {widths}")
    print(f"  Found {len(num_samples)} unique num_samples: {num_samples}")

    if len(widths) < 2 or len(num_samples) < 2:
        print("  Warning: Need at least 2 unique values for both width and num_samples")
        return

    # Create pivot table: loss[width, num_samples]
    loss_grid = df.groupby(['widths', 'n_samples'])['final_test_loss'].mean().unstack()

    def get_loss(w, n):
        """Safely get loss value from grid."""
        try:
            return loss_grid.loc[w, n]
        except KeyError:
            return None

    def calculate_exponent(loss1, loss2, param1, param2):
        """Calculate power-law exponent: m = d(ln(loss)) / d(ln(param))."""
        if loss1 is None or loss2 is None or loss1 <= 0 or loss2 <= 0:
            return None
        return (np.log(loss2) - np.log(loss1)) / (np.log(param2) - np.log(param1))

    # Build LaTeX table rows
    rows = []

    # Header
    header_cols = [f"${num_samples[j]}$" for j in range(len(num_samples) - 1)]
    rows.append("$w$ & " + " & ".join(header_cols) + " \\\\")
    rows.append("\\midrule")

    # Data rows
    for i in range(len(widths) - 1):
        w1, w2 = widths[i], widths[i+1]
        cells = []

        for j in range(len(num_samples) - 1):
            n1, n2 = num_samples[j], num_samples[j+1]

            # Get losses at the four grid points
            loss_w1_n1 = get_loss(w1, n1)
            loss_w2_n1 = get_loss(w2, n1)
            loss_w1_n2 = get_loss(w1, n2)

            # Calculate exponents
            m_w = calculate_exponent(loss_w1_n1, loss_w2_n1, w1, w2)
            m_n = calculate_exponent(loss_w1_n1, loss_w1_n2, n1, n2)

            # Format cell
            if m_w is not None and m_n is not None:
                cells.append(f"{m_w:.2f}, {m_n:.2f}")
            else:
                cells.append("---")

        rows.append(f"${w1}$ & " + " & ".join(cells) + " \\\\")

    # Construct full LaTeX table
    num_cols = len(num_samples) - 1
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Power-law exponents for test loss vs width and num\\_samples. Row headers show starting width $w$, column headers show starting num\\_samples $n$. Each cell $(w, n)$ contains two values: the width exponent (calculated from $w$ to next width at fixed $n$), and the num\\_samples exponent (calculated from $n$ to next num\\_samples at fixed $w$).}}
\\label{{tab:powerlaw_exponents}}
\\begin{{tabular}}{{c{"c" * num_cols}}}
\\toprule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Save to file
    output_path = output_dir / "powerlaw_exponents.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"  Saved: {output_path}")


def generate_best_configurations_table(df, output_dir):
    """
    Generate a table showing the best performing configurations.

    Args:
        df: DataFrame containing results
        output_dir: Directory to save the table
    """
    print("Generating best configurations table...")

    # Get top 10 configurations by final test loss
    top_configs = df.nsmallest(10, 'final_test_loss')

    # Select relevant columns for display
    display_cols = []
    for col in ['widths', 'depths', 'n_samples', 'final_test_loss',
                'final_train_loss', 'best_test_loss', 'training_time_seconds']:
        if col in top_configs.columns:
            display_cols.append(col)

    # Create table
    table_df = top_configs[display_cols].copy()

    # Rename columns for better LaTeX output
    column_mapping = {
        'widths': 'Width',
        'depths': 'Depth',
        'n_samples': 'Samples',
        'final_test_loss': 'Final Test Loss',
        'final_train_loss': 'Final Train Loss',
        'best_test_loss': 'Best Test Loss',
        'training_time_seconds': 'Time (s)'
    }
    table_df.rename(columns=column_mapping, inplace=True)

    # Format columns with appropriate precision
    formatters = {}
    for col in table_df.columns:
        if 'Loss' in col:
            # Use scientific notation for loss columns
            formatters[col] = lambda x: f'{x:.3e}'
        elif col == 'Time (s)':
            # Use 2 decimal places for time
            formatters[col] = lambda x: f'{x:.2f}'

    # Convert to LaTeX with custom formatters
    latex_table = table_df.to_latex(
        index=False,
        formatters=formatters,
        caption="Top 10 configurations by final test loss",
        label="tab:best_configs",
        escape=True
    )

    # Save to file
    output_path = output_dir / "best_configs.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"  Saved: {output_path}")


def generate_tables(csv_path):
    """
    Generate LaTeX tables from experiment results.

    Args:
        csv_path: Path to the results CSV file

    The tables will be saved to ./tables/{expname}/ directory
    """
    # Load the results
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} experiments from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"DataFrame shape: {df.shape}")
    print()

    # Create output directory
    csv_parent = Path(csv_path).parent
    expname = csv_parent.name
    tables_dir = Path('tables') / expname
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tables will be saved to: {tables_dir}/")
    print()

    # Example: Summary statistics table
    generate_summary_table(df, tables_dir)

    # Example: Best configurations table
    generate_best_configurations_table(df, tables_dir)

    # Generate power-law exponents table
    generate_powerlaw_exponents_table(df, tables_dir)

    # Generate master document that includes all tables
    generate_master_document(expname, tables_dir)


def main():
    """Main entry point for the table script."""
    if len(sys.argv) != 2:
        print("Usage: python table.py <path_to_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Generate tables
    generate_tables(csv_path)

    print()
    print("Table generation complete!")


if __name__ == "__main__":
    main()
