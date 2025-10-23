#!/bin/bash
# Automated experiment workflow script
#
# This script runs the complete experiment pipeline:
# 1. Generate parameter files
# 2. Run all experiments in parallel
# 3. Collate results
# 4. Generate plots

set -e  # Exit on error

# Extract experiment name from makeparams.py
EXPNAME=$(python -c "
import sys
sys.path.insert(0, '.')
from makeparams import params
print(params['name'])
")

echo "==============================================="
echo "Running experiment: $EXPNAME"
echo "==============================================="
echo

# Step 1: Generate parameter files
echo "[Step 1/4] Generating parameter files..."
python makeparams.py
echo "  Parameter files generated in params/$EXPNAME/"
echo

# Step 2: Run all experiments in parallel
echo "[Step 2/4] Running experiments in parallel..."
parallel --bar --line-buffer --joblog progress.log python runner.py ::: params/$EXPNAME/*.yaml
echo "  All experiments completed"
echo

# Step 3: Collate results
echo "[Step 3/5] Collating results..."
python collate.py results/$EXPNAME
echo "  Results collated to results/$EXPNAME/results.csv"
echo

# Step 4: Generate tables
echo "[Step 4/5] Generating LaTeX tables..."
python table.py results/$EXPNAME/results.csv
echo "  Tables saved to tables/$EXPNAME/"
echo

# Step 5: Generate plots
echo "[Step 5/5] Generating plots..."
python plot.py results/$EXPNAME
echo "  Plots saved to plots/$EXPNAME/"
echo

echo "==============================================="
echo "Experiment $EXPNAME completed successfully!"
echo "==============================================="
echo "Results: results/$EXPNAME/results.csv"
echo "Tables: tables/$EXPNAME/"
echo "  - tables.tex (master document)"
echo "  - summary_stats.tex"
echo "  - best_configs.tex"
echo "Plots: plots/$EXPNAME/"
echo "  - testloss.pdf (surface plot)"
echo "  - history_*.pdf (training history plots)"
echo "Job log: progress.log"
