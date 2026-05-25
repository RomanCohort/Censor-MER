# Censor-MER 训练与实验日志

## 2026-05-25

### 微表情生成器训练

#### v6 生成器
- **路径**: `checkpoints/censor_g_gen_v6/censor_g_gen_final.pth`
- **SSIM**: 0.82-0.83
- **特点**: 强制运动放大（motion_scale=50）

#### RLHF优化
- **路径**: `checkpoints/rlhf_gen/rlhf_final.pth`
- **Epoch**: 20
- **Reward**: 0.2
- **方法**: Policy Gradient，识别器作为奖励模型

### 识别器性能
- **Cross-domain CASME2**: 87% (cross_src_casme2_best.pth)
- **路径**: `checkpoints/cross_casme2/cross_src_casme2_best.pth`

### SNN实验
- **结果文件**: `results/snn_experiment_results.json`
- **发现**: SNN在可解释性方面优于ANN，但时序性能较低

### 关键改进
1. 修复SMIC加载（支持micro/positive/negative结构）
2. 添加运动约束损失（motion_loss + temporal_var_loss）
3. 强制最小运动幅度（5像素）
4. 使用最后一帧作为中性脸（避免复制shortcut）
5. RLHF训练框架（识别器作为奖励）

### 下一步
- [ ] 提升RLHF reward
- [ ] Web界面收集人类反馈
- [ ] 更多样化的表情生成测试