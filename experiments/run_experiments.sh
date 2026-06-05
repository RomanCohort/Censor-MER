#!/bin/bash
# =============================================================================
# AutoDL Experiment Runner
# =============================================================================
# Usage:
#   bash run_experiments.sh        # Run all experiments
#   bash run_experiments.sh 1      # Run experiment 1 only
#   bash run_experiments.sh 2 3    # Run experiments 2 and 3
# =============================================================================

set -e  # Exit on error

# Activate conda environment (modify as needed)
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate censor

# Create results directory
mkdir -p results

# Get experiments to run
EXPS=${@:-"1 2 3"}

echo "=============================================="
echo "Censor - AutoDL Experiments"
echo "=============================================="
echo "Experiments to run: $EXPS"
echo "Start time: $(date)"
echo "=============================================="

# Run experiments
for exp in $EXPS; do
    echo ""
    echo ">>> Running Experiment $exp <<<"
    echo ""

    case $exp in
        1)
            echo "OFF-ApexNet LOSO Reproduction"
            python experiments/autodl_experiments.py --exp 1
            ;;
        2)
            echo "rPPG Signal Quality Validation"
            python experiments/autodl_experiments.py --exp 2
            ;;
        3)
            echo "Inference Latency Benchmark"
            python experiments/autodl_experiments.py --exp 3
            ;;
        *)
            echo "Unknown experiment: $exp"
            ;;
    esac
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "Results saved to: results/"
echo "=============================================="

# List results
ls -la results/
