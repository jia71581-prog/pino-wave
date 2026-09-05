# Transfer DG-WFP E1d-v2 四族泛化结论

日期：2026-09-02

## 判定

E1d-v2 通过预注册的 train-only group-disjoint confirmation 门槛，应进入全部 2800 条 train 派生标签生成与正式预训练。

这只是架构方向通过。phase confirmation 均值仍为 0.6514，远未达到最终 0.05 精度要求，不能晋升为预训练模型。

## 数据与访问边界

- 256 条 train，256 个互异 group。
- 每族 48 fit、8 calibration、8 confirmation。
- 训练使用完整 401 帧和全部派生标签，符合“训练数据所有信息均可用”。
- 完整波场只作为目标与固定能量分母，模型输入未来波场帧数为 0。
- 最终 validation/test 仍限定为速度、震源、边界、eikonal 和注册的早期快照前缀；后续真值不可作为输入或在线调参信息。
- validation/test_id 本轮均未读取。

## 走时修正

原 radius-4 grid-eikonal 在 uniform 上有 9.7--12.4 ms 方向性误差，因此训练前被拒绝。v2 使用：

`T = T_graph(c) - T_graph(c_source) + distance_exact/c_source`。

这消除相同网格的常速方向误差，同时保留非均匀速度造成的走时扰动。256 条 v2 走时全部有限、能量分母全部为正，uniform 最大误差为 0。

## 训练

- raw WFP 与 phase WFP 各两个种子。
- width 32、rank 16、depth 4，124,556 参数。
- 5 epochs，每 epoch 3072 updates，总计 15,360 updates。
- 每个 epoch 对所有 192 条 fit 记录和全部 64 个时间频率精确覆盖一次。
- 左/右/下 20 层外部 CPML，顶部自由表面。

## 独立 confirmation

| arm | seed 372 | seed 733 | 两种子均值 |
|---|---:|---:|---:|
| raw | 0.990196 | 0.989921 | 0.990058 |
| phase | 0.652863 | 0.649941 | 0.651402 |

phase 相对 raw 改善 34.21%。

按 family 的两种子均值：

| family | raw | phase |
|---|---:|---:|
| uniform | 0.990376 | 0.611188 |
| layered | 0.984718 | 0.634567 |
| anomaly | 0.989748 | 0.655310 |
| marmousi | 0.995392 | 0.704542 |

四族全部改善。phase 两种子的 confirmation cosine 为 0.7649/0.7666，预测/真值能量比为 0.7594/0.7608；raw cosine 仅约 0.15。收益来自相位与形状对齐，不只是幅值放大。

CPML 辅助误差仍为约 0.86，说明外部状态学习较弱；但 top/outer pressure 硬边界违规均为 0。

## 下一步

1. 为全部 2800 条 train 生成 sigma=2 背景、64-bin 系数、20 层外部 CPML 标签和 v2 去偏 eikonal。
2. 先按 group 做 fit/calibration/confirmation 训练与选择。
3. 结构固定后，用全部 2800 条重新训练最终预训练模型。
4. 在全量阶段增加模型容量、频率 progressive opening 和 DG 界面分支，目标是从当前 0.65 继续降低，而不是把 E1d 小模型当成最终网络。
