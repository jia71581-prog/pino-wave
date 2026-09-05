# V5 因果实例化微调：三类介质验证结果

## 验证范围

- 数据集：`/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5`
- 父模型：`artifacts/v5_instance_adaptation/parent/best.pt`（当前最优完整验证 checkpoint，epoch 5）
- 每类介质 1 个验证实例：Uniform、Layered、Marmousi
- 每个实例只读取起振后的相邻两帧真值；本次实例的审计索引分别为 `[45,46]`、`[71,72]`、`[49,50]`
- 微调：12 步 AdamW + 4 步固定闭包 L-BFGS，低秩空间残差 + 起振快照 conditioner；候选必须同时通过观测、PDE 残差和能量门控
- 评估：完整 401 个保存时间步的波场相对 L2，以及由完整波场派生的 9 个接收器波形相对 L2

## 结果

| 介质 | 样本 | 适配是否接受 | 回滚原因 | 完整波场相对 L2 | 接收器波形相对 L2 |
|---|---|---:|---|---:|---:|
| Uniform | `validation_uniform_00069` | 否 | 起振帧观测损失未改善 | 0.696954 | 0.289122 |
| Layered | `validation_layered_00048` | 否 | 起振帧观测损失未改善 | 0.501110 | 0.263884 |
| Marmousi | `validation_marmousi_00109` | 否 | 起振帧观测损失未改善 | 0.723463 | 0.322003 |

三例均通过安全回滚，故最终预测为父模型预测；这不是把未通过门控的候选结果当成改进。当前结果尚未达到 10% 误差目标，说明仅凭两个早期快照，当前父模型和轻量适配器仍不足以恢复长时高频传播。

## 输出图件

每个实例都生成了包含真值、父模型、适配模型及误差图的波场快照（起振后早期帧、中间帧、末帧），以及接收器波形对比：

- `artifacts/v5_instance_adaptation/pilot_final2/validation_uniform_00069/wavefield_comparison.png`
- `artifacts/v5_instance_adaptation/pilot_final2/validation_uniform_00069/receiver_waveforms.png`
- `artifacts/v5_instance_adaptation/pilot_final2/validation_layered_00048/wavefield_comparison.png`
- `artifacts/v5_instance_adaptation/pilot_final2/validation_layered_00048/receiver_waveforms.png`
- `artifacts/v5_instance_adaptation/pilot_final2/validation_marmousi_00109/wavefield_comparison.png`
- `artifacts/v5_instance_adaptation/pilot_final2/validation_marmousi_00109/receiver_waveforms.png`

PDF 矢量版本位于同名目录下的 `.pdf` 文件。`fields.pt` 保存了每个实例完整 401 步的父模型和最终（门控后）波场预测。

## 可复现性与安全性

- `evaluation.json` 在打开后续真值之前写入并封存适配状态；适配阶段的 `future_truth_used` 为 `false`。
- 适配状态以张量形状和 SHA-256 摘要序列化，避免把大张量直接写入 JSON。
- PDE 残差使用压力/`dt²` 的独立尺度归一化，并对残差输出做有界化，避免适配时数值爆炸。
- 相关测试：`tests/saved_time_phase_operator_v4` 全部通过。

## 后续研究结论

当前失败不是继续增大学习率即可解决：Uniform 的候选 PDE 残差从 `1.16e6` 增至 `5.11e7`，而 Layered/Marmousi 的候选在观测门控上也未改善。下一阶段应先离线训练起振 conditioner（用训练集实例的完整真值，仅训练阶段读取未来帧），再冻结 conditioner 做真正的零样本实例适配；同时把桥接状态从 4 步显式推进扩展为可学习的因果状态，而不是放宽验证阶段的两帧真值约束。

## Conditioner 后续阶段结果

已完成 128 个 train-split episode 的离线 conditioner 训练，checkpoint 为：

`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/v5_instance_adaptation/conditioner.pt`

训练辅助损失从约 `0.1975` 降至末段约 `0.0045`。验证适配阶段加载该 checkpoint，并再次运行相同三个实例；适配仍全部触发观测门控回滚：Uniform、Layered、Marmousi 的完整波场误差和接收器误差分别保持为 `69.70%/28.91%`、`50.11%/26.39%`、`72.35%/32.20%`。

这表明当前 summary-head conditioner 学到的是未来场统计量，而不是能直接校正父模型时空相位的 latent。下一步应改为训练“早期两帧 → 低秩时空残差参数”的监督目标，并在训练阶段对残差的观测、PDE 和能量门控联合优化；当前结果不应宣称已达到 10% 误差。

本次加载 conditioner 的验证输出位于：

`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/v5_instance_adaptation/pilot_conditioned/`

## 低秩残差元预训练结果

随后完成了 12 个 train-split episode 的“前两帧 → 低秩时空残差”元预训练，checkpoint 为：

`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/v5_instance_adaptation/meta_pretrained.pt`

训练时使用 50 个抽样时间点监督完整场残差，后续验证仍只读取两个起振帧。初始试验发现预训练残差可能本身恶化 Uniform，因此已将门控修正为始终以冻结父模型作为 baseline；失败时恢复零残差，而不是恢复可能有害的预训练残差。

使用 epoch 5 父模型的安全验证结果为：Uniform `69.70%/28.91%`、Layered `50.11%/26.39%`、Marmousi `72.35%/32.20%`（波场/接收器），三例均被安全回滚。最终图件位于：

`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/v5_instance_adaptation/pilot_meta_epoch5_safe/`

## PID 2630329 长程预训练复评

PID `2630329` 仍在运行，当前已完成 epoch 16；训练期间 GPU 利用率约 100%，功率约 334–350 W。训练面板的最低 aggregate relative L2 为 epoch 12 的 `0.530881`，但独立完整 401 步三实例评估显示实际 `best.pt`（epoch 15）为：

| checkpoint | Uniform | Layered | Marmousi |
|---|---:|---:|---:|
| epoch 15 best：波场相对 L2 | 0.630982 | 0.490666 | 0.730697 |
| epoch 16 latest：波场相对 L2 | 0.642174 | 0.479258 | 0.742304 |

对应接收器波形相对 L2 分别为 epoch 15 的 `0.215562/0.198843/0.277940` 和 epoch 16 的 `0.195837/0.210163/0.284376`（Uniform/Layered/Marmousi）。因此 epoch 15 暂作为独立评估 best；epoch 16 只在 Layered 上改善，尚不能认为整体优于 epoch 15。

复评输出：

- `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/v5_instance_adaptation/parent_epoch15_eval/`
- `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/v5_instance_adaptation/parent_epoch16_latest_eval/`
- `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/v5_instance_adaptation/parent_epoch15_best.pt`
- `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/v5_instance_adaptation/parent_epoch16_latest.pt`
