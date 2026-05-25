#!/bin/bash
# =============================================================================
# AutoDL Scripts: Censor-G SNN Generation Pipeline
# =============================================================================
# 使用真实CASME2数据进行生成评估
#
# 步骤：
#   1. 数据准备
#   2. SNN训练（30 epochs）
#   3. EventDriven训练（30 epochs）
#   4. CASME2生成评估
#   5. 论文数据导出
# =============================================================================

# 目录设置
ROOT="/root/autodl-tmp/censor"
DATA_ROOT="/root/autodl-tmp/data/CASME2"
SAVE_DIR="$ROOT/results/generation_casme2"

mkdir -p $SAVE_DIR
mkdir -p $ROOT/logs

# =============================================================================
# Step 1: 数据准备
# =============================================================================
echo "============================================================"
echo "[Step 1] Data Preparation"
echo "============================================================"

# 检查CASME2数据
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: CASME2 not found at $DATA_ROOT"
    echo "Please download CASME2 dataset first"
    exit 1
fi

# 数据统计
echo "  CASME2 path: $DATA_ROOT"
ls -la $DATA_ROOT | head -10

# =============================================================================
# Step 2: SNN训练（真实数据）
# =============================================================================
echo "============================================================"
echo "[Step 2] SNN Training with Real CASME2"
echo "============================================================"

python $ROOT/generation/train_snn.py \
    --dataset casme2 \
    --data_root $DATA_ROOT \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-4 \
    --spike_threshold 1.0 \
    --spike_time_steps 10 \
    --analyze_spikes \
    --save_dir $ROOT/checkpoints/censor_g_snn_casme2 \
    --log_dir $ROOT/logs/snn_casme2 \
    --save_every 5

echo "  SNN training complete!"

# =============================================================================
# Step 3: EventDriven训练
# =============================================================================
echo "============================================================"
echo "[Step 3] EventDriven Training"
echo "============================================================"

python $ROOT/generation/train_snn.py \
    --dataset casme2 \
    --data_root $DATA_ROOT \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-4 \
    --train_mode all \
    --analyze_spikes \
    --save_dir $ROOT/checkpoints/censor_g_snn_event_casme2 \
    --log_dir $ROOT/logs/snn_event_casme2

echo "  EventDriven training complete!"

# =============================================================================
# Step 4: CASME2生成评估
# =============================================================================
echo "============================================================"
echo "[Step 4] CASME2 Generation Evaluation"
echo "============================================================"

python $ROOT/experiments/casme2_generation_eval.py \
    --data_root $DATA_ROOT \
    --num_samples 100 \
    --snn_checkpoint $ROOT/checkpoints/censor_g_snn_casme2/censor_g_snn_best.pth \
    --event_checkpoint $ROOT/checkpoints/censor_g_snn_event_casme2/censor_g_snn_best.pth \
    --results_dir $SAVE_DIR \
    --num_frames 16 \
    --image_size 224

echo "  Generation evaluation complete!"

# =============================================================================
# Step 5: 导出论文数据
# =============================================================================
echo "============================================================"
echo "[Step 5] Export Paper Data"
echo "============================================================"

# 汇总结果
python -c "
import json
import os

# 加载评估结果
eval_path = '$SAVE_DIR/generation_evaluation.json'
if os.path.exists(eval_path):
    with open(eval_path) as f:
        results = json.load(f)

    print('\\nPaper Results Summary:')
    print('='*50)

    if 'snn' in results.get('summary', {}):
        snn = results['summary']['snn']
        print(f'  SNN FID: {snn.get(\"fid_mean\", \"N/A\"):.4f}')
        print(f'  SNN Temporal: {snn.get(\"temporal_mean\", \"N/A\"):.4f}')

    if 'event' in results.get('summary', {}):
        event = results['summary']['event']
        print(f'  EventDriven FID: {event.get(\"fid_mean\", \"N/A\"):.4f}')
        print(f'  Events/sample: {event.get(\"events_mean\", \"N/A\"):.1f}')

    # 按情感分析
    if 'emotion_summary' in results:
        print('\\n  By Emotion:')
        for emotion, metrics in results['emotion_summary'].items():
            print(f'    {emotion}: FID={metrics.get(\"fid_mean\", \"N/A\"):.4f}')
"

# 保存到论文目录
cp $SAVE_DIR/generation_evaluation.json $ROOT/paper/snn_paper/data/

echo "  Paper data exported!"

# =============================================================================
# 完成
# =============================================================================
echo "============================================================"
echo "[Complete] All steps finished!"
echo "============================================================"

echo ""
echo "Results saved to:"
echo "  $SAVE_DIR/"
echo ""
echo "Checkpoints saved to:"
echo "  $ROOT/checkpoints/censor_g_snn_casme2/"
echo "  $ROOT/checkpoints/censor_g_snn_event_casme2/"
echo ""
echo "Paper data:"
echo "  $ROOT/paper/snn_paper/data/generation_evaluation.json"