# Elastic VTI 分量均衡损失设计

## 目标

降低 Marmousi 切片 Elastic VTI FNO 的 `u_x` 预测误差，同时保护已经较准确的 `u_z` 分量。方法只改变监督 relative L2 的分量归一化方式，复用 epoch 132 的原始验证最优权重，不改变数据、模型结构或物理损失。

## 根因

当前监督损失对形状 `[B,H,W,T,2]` 的两个位移分量整体计算 relative L2。四样本评估中：

- `u_x` target norm 为 `0.7067`，relative L2 为 `0.1385`；
- `u_z` target norm 为 `1.1673`，relative L2 为 `0.0848`；
- `u_x` 只占两个分量目标平方能量约 26.8%。

整体归一化使高能量 `u_z` 主导监督梯度。继续调整 AdamW 学习率无法改变这一目标函数偏置，三学习率筛选也未达到预注册精度门槛。

## 损失定义

对每个 batch 样本和每个分量分别计算：

```text
L_c = ||prediction_c - target_c||_2 / max(||target_c||_2, eps)
L_component = mean_batch(sum_c(w_c * L_c) / sum_c(w_c))
```

第一阶段固定 `w_x = 1`、`w_z = 1`。权重必须为长度 2 的有限正数列表。未配置 `component_relative_l2_weights` 时保持现有整体 relative L2 行为，确保其他声学和弹性配置不变。

MSE、PDE、energy 和 receiver 损失继续使用原定义。分量均衡只替换监督损失中的 relative L2 项，不重复添加原整体 relative L2。

## 实现边界

- 在 Elastic VTI 训练模块内实现分量均衡 helper，不改变声学 `combined_loss` 的默认接口。
- helper 要求预测与目标形状相同、最后一维为组件维。
- 将 `component_relative_l2_weights` 加入 Elastic VTI 监督配置白名单。
- 普通训练与空间重要性采样路径必须使用同一分量定义；本次候选关闭空间重要性采样，但测试覆盖该接口的形状和重加权边界。
- 日志记录 `relative_l2_component_0`、`relative_l2_component_1` 和加权均值。
- 验证模型选择仍使用现有全局 relative L2，避免改变历史 checkpoint 的比较口径。

## 短程候选

从以下冻结基线初始化：

- checkpoint：`artifacts/elastic_vti_pino_marmousi/checkpoints/best.pt`；
- SHA256：`7f05716e87075718df9291ca76aa5aaacb85e071c6e35d819d5ef4762c54efd8`；
- epoch 132；
- 全局验证 relative L2：`0.1049973499`。

候选参数：

- 5 epochs；
- 每 epoch 最多 64 batches；
- AdamW learning rate `2e-5`；
- weight decay `1e-5`；
- grad clip `0.5`；
- batch size 1；
- 新 optimizer 和 cosine scheduler，禁止 `--resume`；
- 固定 split、归一化、seed 和最多 8 个验证 batches。

候选写入独立 artifact 目录，不覆盖基线或之前的优化器 sweep。

## 验收与回退

短程候选封存后，使用与基线相同的验证样本 `[3,23,40,48]` 做组件评估。必须同时满足：

- `u_x` relative L2 `<= 0.13161056`，相对基线至少改善 5%；
- `u_z` relative L2 `<= 0.08560451`，相对基线恶化不超过 1%；
- 全局验证 relative L2 `<= 0.1049973499`；
- 预测、损失和梯度有限；
- 无 OOM，checkpoint 可重新读取；
- 评估前不读取测试集。

任一条件失败时保留原始 `best.pt`，不提交正式训练。全部通过后才生成正式 40-epoch 配置并以 `nohup` 提交。

## 非目标

- 不显式提高 `u_x` 权重到 1.5 或 2；
- 不扩大 modes、width 或层数；
- 不改变 PDE、energy、receiver 权重；
- 不重新划分数据或更新归一化统计；
- 不承诺损失调整必然突破模型容量上限。

