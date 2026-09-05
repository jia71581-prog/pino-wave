# TGRS 实验结果证据包

本文件夹集中整理当前声学算子研究的图、逐样本数据、网络超参数、实验协议和
可复现性审计。大型完整波场采用相对符号链接，避免重复占用约 440 MB；其余
图表和报告均实体复制。先查看 `EVIDENCE_STATUS.csv`，其中明确区分历史开发
证据、训练集选模证据、完整确认实验和等待中的实验。

## 内容导航

- `01_complete_wavefields`：Uniform、Layered、Marmousi 各一个历史开发样本的
  401 帧真值、粗解和预测数组。
- `02_wavefield_snapshots`：完整波场快照以及参考、预测、误差对比图。
- `03_receiver_waveforms`：接收器时间波形与浅层接收线炮集图。
- `04_numerical_dispersion`：FD2、FD4、LWC-84 的解析相速度误差 PDF/PNG/CSV，
  以及 learned-dispersion claim gate。当前只能声称量化了数值频散，不能声称
  神经网络已经克服或消除了频散。
- `05_relative_error`：两套完整数据。第一套是 r5b 固定训练门控全部 48 条记录；
  第二套是 CPADC R7 全部 480 条 validation 和全部 480 条独立 `test_id`，包含
  960 行配对 CSV、箱线图 PDF/PNG 和原始 summary JSON。
- `06_network_hyperparameters`：r5b、r5d/r5e、参数匹配 Patch-DeepONet 的
  超参数 JSON/CSV/Markdown，以及原始 YAML 配置。
- `07_protocols_and_figure_plan`：固定 19 Hz、仅震源位置泛化的冻结协议和图版计划。
- `08_reproducibility_and_diagnostics`：相对误差平台期、selector seed、VDS 文件依赖、
  网络结构和 Patch-DeepONet 审计。
- `09_pending_position_evaluation`：30 个未见 Marmousi 切片乘 8 个位置的完整评估。
  为保持历史路径稳定而保留旧目录名，其中包含预测/真值封存清单、240 条逐样本
  指标、快照、接收器炮集和箱线图。
- `10_training_runs_read_only`：r5b/r5c/r5d 规范训练目录的只读链接，便于检查训练曲线、
  retry 记录和 checkpoint 元数据。
- `12_phase4b_fixed19_rank15_multisource`：Phase4b 在 rank-15 真实 Marmousi
  速度切片上的固定 19 Hz 八震源合成图、预测封存清单、逐位置指标和可复现脚本。
  这是单切片案例，与 G3 Marmousi 的 4.51% 结果严格分开。
- `13_phase4b_rank30_eight_source_superposition`：按输入速度复杂度选择的 rank-30
  Marmousi 切片，以及八个固定 19 Hz 单源场未经缩放的物理叠加。该图用于展示
  复杂介质中的多界面反射和散射，不作为精度晋级证据。
- `14_runtime_comparison`：我们的方法与 LWC-84 的可复现耗时汇总。“我们的方法”
  包含绑定父算子推理、实例系数微调、反归一化和 401 帧输出物化。约 2 倍加速
  仅为描述性结果，因为存档实验没有相同样本配对，也没有达到相同绝对精度。
- `15_pi_deeponet_train_development_comparison`：DFO+RCFA 与全量训练
  PI-DeepONet 在六条相同训练记录、相同未来窗口上的开发集对比和原始快照数据。
- `16_phase4b_update0265_pi_comparison`：历史最优 Phase4b update 265 与
  PI-DeepONet 在上述六条记录上的重评。逐记录均值为 `0.105992` 对 `0.615635`。
  Phase4b 明确包含外部 sigma-2 LWC-84 背景，且此结果没有应用实例适配器。
- `99_historical_experiment_index`：`results/` 下每个顶层实验的一层注册表。大型历史
  checkpoint 仅登记路径，不递归复制。

## 当前可引用的数值

- r5b 固定 48 条训练门控平均相对误差为 0.277254；Uniform、Layered、Marmousi
  分别为 0.108413、0.252105、0.573991。该结果仅用于训练集选模诊断。
- CPADC R7 在完整 validation 上平均相对改善 2.7006%，97.2917% 的记录不变差；
  在独立 `test_id` 上平均改善 2.8420%，97.0833% 的记录不变差。它支持受限的
  相对校正收益，不支持绝对 5% 求解器误差声明。
- 我们的方法在独立 `test_id` 上的实例微调和推理物化均值分别为 0.452844 秒和
  9.957606 秒，端到端均值为 10.410450 秒。LWC-84 同步基准均值为 20.499601 秒，均值加速比为
  1.969137。微调仅占总耗时的 4.35%；即使完全消除微调，均值加速也只有
  2.058688 倍。该对比没有通过同样本、同精度或 10 倍加速门槛。
- 在每波长 4 个网格点处，FD2、FD4、LWC-84 的最大解析相速度误差分别为
  9.49%、2.37%、0.337%。不得写成“无频散”。

## 论文图组织原则

主文候选应包括固定 19 Hz 多位置结果、接收器相位/延迟、完整样本误差分布和
解析频散背景；单样本历史快照、早期 adaptation 图以及失败基线放补充材料。
每个图注必须注明 split、样本数、震源频率以及它属于选模还是确认实验。

本研究的震源泛化变量只允许是位置，频率固定为 19 Hz，不允许频率泛化结论。

## 最新候选模型测试已完成（2026-08-13）

固定 19 Hz、仅改变震源位置的 240 条测试已完成，完整瞬态相对 L2 均值为 `0.581776439647`。与 Phase4b 的三类同样本严格替换门槛 全部未通过，因此没有替换历史优胜模型资产。完整结果见 `09_pending_position_evaluation`（为保持路径稳定而保留的旧目录名）和 `11_latest_candidate_nonpromotion_test`。

## Phase4b 多震源案例（2026-08-14）

update 265 已在预注册的 rank-15 速度切片和八个固定 19 Hz 震源位置上完成推理。
八个全 401 帧误差平均为 `0.176883`，训练位置范围内为 `0.177279`，范围外为
`0.176224`。最终合成图包含真实速度模型。单独的 G3 Marmousi 最优记录仍为
`0.045074`（4.51%）；两者都依赖外部平滑背景数值求解。
