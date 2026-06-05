#!/bin/bash
# =============================================================================
# Run All Experiments (Parallel Version)
# =============================================================================
# Usage:
#   bash experiments/run_all.sh          # Run all 4 experiments
#   bash experiments/run_all.sh 1 3 4    # Run specific experiments
#
# NOTE: Exp 2 (rPPG validation) removed - ME time window too short
# =============================================================================

EXPS=${@:-"1 3 4 5"}
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
        *)
            echo "Unknown: $exp (valid: 1, 3, 4, 5)"
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
ps aux | grep "exp[1-5]" | grep -v grep