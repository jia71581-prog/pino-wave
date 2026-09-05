# Transfer DG-WFP E1c 相位载波容量结论

日期：2026-09-02

## 判定

E1c 通过预注册的单记录容量门槛。应进入下一层 256 条四族 group-disjoint 泛化实验，但不能将当前结果视为预训练模型晋升。

## 方法

- 数据：E1 缓存中的 `train_uniform_00010`，64 个时间 rFFT bin。
- 网络：相同的 124,556 参数 WFP。
- control：直接输出 raw complex coefficient。
- candidate：输出复振幅，再乘部署可得的精确 uniform 走时载波 `exp(-i*omega*tau)`。
- loss：完整记录能量作为固定分母；每 16 updates 无重复覆盖全部 64 个频率。
- 外部边界：左/右/下 20 层 CPML，顶部自由表面。
- 四卡：raw/phase 各两个种子，800 updates。

## 结果

| arm | seed | best relative L2 | pred/target norm | cosine |
|---|---:|---:|---:|---:|
| raw | 372 | 0.972189 | 0.19337 | 0.23851 |
| raw | 733 | 0.969778 | 0.19080 | 0.25140 |
| phase | 372 | 0.700907 | 0.50113 | 0.75815 |
| phase | 733 | 0.674754 | 0.55931 | 0.76660 |

- raw 两种子均值：0.970984。
- phase 两种子均值：0.687830。
- phase 相对改善：29.16%。
- 两个 phase 种子均低于 0.95 容量门槛，并且预测能量比大于 0.05。
- top/outer pressure 最大违规均为 0。

## 解释

相位载波解决了 E1/E1b 的主要零解问题。它把网络的任务从“凭空生成全域快速振荡相位”改为“在已知传播相位上学习较平滑的复振幅和有限窗残差”。相关性从 raw 的约 0.24 提高到约 0.76，说明收益不是单纯输出幅值变大。

绝对误差仍为 0.67--0.70，因此当前结果只足以支持下一层实验。它还不能证明：

- 非均匀平滑速度的 eikonal 走时足够准确；
- layered/anomaly/Marmousi 的多到达和晚期反射能泛化；
- 256 条训练后能超越已有预训练模型；
- 最终速度和 5% 精度目标已经达到。

## 下一步

E1d 使用现有 256 条缓存，不新增波动方程数据：

1. 对每条 sigma=2 平滑速度计算 source-dependent grid-eikonal travel time；
2. raw/phase 两种子继续配对，使用稳定的完整记录能量分母；
3. calibration 选 checkpoint，独立 confirmation 比较；
4. 只有 phase confirmation 严格改善，才重新考虑 2800 条扩展。

validation/test_id 继续不读取。
