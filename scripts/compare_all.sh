#!/bin/bash
# 对比测试：评估所有模型

cd /root/autodl-tmp/Censor-MER

echo "============================================================"
echo "Comparative Testing: Hybrid vs GAN vs RLHF"
echo "============================================================"

# 测试Hybrid
echo ""
echo "[Testing Hybrid Model]"
python generation/test_hybrid.py \
    --checkpoint "./checkpoints/hybrid_model_v2/hybrid_final.pth" \
    --output_dir "./results/compare_hybrid" \
    --num_samples 20 \
    > logs/compare_hybrid.log 2>&1

# 测试GAN
echo ""
echo "[Testing GAN Model]"
python generation/test_hybrid.py \
    --checkpoint "./checkpoints/gan_generator/gan_best.pth" \
    --output_dir "./results/compare_gan" \
    --num_samples 20 \
    > logs/compare_gan.log 2>&1

# 测试RLHF
echo ""
echo "[Testing RLHF Model]"
python generation/test_hybrid.py \
    --checkpoint "./checkpoints/rlhf_gen_v2/rlhf_best.pth" \
    --output_dir "./results/compare_rlhf" \
    --num_samples 20 \
    > logs/compare_rlhf.log 2>&1

# 显示结果
echo ""
echo "============================================================"
echo "[Results Summary]"
echo "============================================================"

echo ""
echo "Hybrid Model:"
cat results/compare_hybrid/test_results.json 2>/dev/null | grep -E "accuracy|confidence" || echo "  Not ready"

echo ""
echo "GAN Model:"
cat results/compare_gan/test_results.json 2>/dev/null | grep -E "accuracy|confidence" || echo "  Not ready"

echo ""
echo "RLHF Model:"
cat results/compare_rlhf/test_results.json 2>/dev/null | grep -E "accuracy|confidence" || echo "  Not ready"

echo ""
echo "============================================================"