#!/bin/bash
# =============================================================================
# Run All Experiments (Parallel Version)
# =============================================================================
# Based on reviewer feedback, run these experiments:
#
# Exp 1: OFF-ApexNet reproduction (R2 - fair SOTA comparison)
# Exp 3: Latency benchmark (deployment guidance)
# Exp 4: Sparse control statistics (parameter efficiency)
# Exp 5: Training curves + Expert specialization (R1, R2)
# Exp 6: MoE expert count ablation (R1 - justify E=3)
#
# Usage:
#   bash experiments/run_all.sh          # Run all 5 experiments
#   bash experiments/run_all.sh 1 5 6    # Run specific experiments
# =============================================================================

EXPS=${@:-"1 3 4 5 6"}
LOG_DIR="results/logs"
mkdir -p $LOG_DIR

echo "=============================================="
echo "Censor - Parallel Experiments"
echo "=============================================="
echo "Experiments: $EXPS"
echo "Logs: $LOG_DIR"
echo "=============================================="

for exp in $EXPS; do
    LOG_FILE="$LOG_DIR/exp${exp}_$(date +%Y%m%d_%H%M%S).log"

    case $exp in
        1)
            echo ">>> Exp 1: OFF-ApexNet (background)"
            nohup python experiments/exp1_offapexnet.py > $LOG_FILE 2>&1 &
            ;;
        3)
            echo ">>> Exp 3: Latency Benchmark (background)"
            nohup python experiments/exp3_latency.py > $LOG_FILE 2>&1 &
            ;;
        4)
            echo ">>> Exp 4: Sparse Control (background)"
            nohup python experiments/exp4_sparse_control.py > $LOG_FILE 2>&1 &
            ;;
        5)
            echo ">>> Exp 5: Training Curves (background)"
            nohup python experiments/exp5_training_curves.py > $LOG_FILE 2>&1 &
            ;;
        6)
            echo ">>> Exp 6: MoE Expert Ablation (background)"
            nohup python experiments/exp6_moe_ablation.py > $LOG_FILE 2>&1 &
            ;;
        *)
            echo "Unknown: $exp (valid: 1, 3, 4, 5, 6)"
            ;;
    esac
    echo "    Log: $LOG_FILE"
done

echo ""
echo "=============================================="
echo "All experiments started in background!"
echo "Check logs in: $LOG_DIR"
echo "=============================================="

# Show running processes
sleep 2
ps aux | grep "exp[1-6]" | grep -v grep