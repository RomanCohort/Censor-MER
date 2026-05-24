"""
Censor 演示脚本

运行方式:
    python demo.py
"""

import torch
import sys
sys.path.insert(0, 'D:/censor')

from main import Censor


def main():
    print("=" * 60)
    print("  Censor - 仿生双通道微表情识别系统演示")
    print("=" * 60)

    # 创建模型
    print("\n[1/4] 加载模型...")
    model = Censor()
    model.eval()

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量: {total_params:,}")

    # 伪造输入
    print("\n[2/4] 创建输入 (伪造视频)...")
    B, C, T, H, W = 2, 3, 16, 224, 224
    dummy_input = torch.randn(B, C, T, H, W)
    print(f"      输入: {dummy_input.shape}")

    # 前向传播
    print("\n[3/4] 前向传播...")
    with torch.no_grad():
        outputs = model(dummy_input)

    # 结果
    print("\n[4/4] 结果:")
    print("-" * 40)

    # 1. 微表情分类
    me_pred = outputs['me_logits'][0].argmax().item()
    me_names = ['Joy', 'Sadness', 'Fear', 'Anger', 'Surprise', 'Disgust', 'Contempt']
    print(f"  微表情预测: {me_names[me_pred]}")

    # 2. AU强度
    au_pred = (outputs['au_intensities'][0] > 0.5).nonzero().squeeze().tolist()
    print(f"  活跃AU: {au_pred[:5]}...")

    # 3. 专家门控
    gate = outputs['expert_gates'][0].numpy()
    print(f"  专家分布: [{gate[0]:.2f}, {gate[1]:.2f}, {gate[2]:.2f}]")

    # 4. Apex帧
    apex = outputs['apex_scores'][0].item()
    print(f"  Apex帧位置: t={apex:.1f}")

    # 5. 报告
    print("\n  模板报告:")
    print(f"    {outputs['template_report'][0][:80]}...")

    print("\n" + "=" * 60)
    print("  演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()