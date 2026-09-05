# Elastic VTI Marmousi 优化器精调设计

## 目标

在不改变数据、网络结构和损失定义的前提下，为 Elastic VTI Marmousi 切片模型选择更适合后期精调的 AdamW 参数，使验证 relative L2 稳定低于当前最优值 `0.1055667941`。任何候选未达到接受门槛时，继续保留当前 `best.pt`，不得把测试集用于选择超参数。

## 当前证据与根因

- 当前模型在 epoch 125 达到最佳验证 relative L2 `0.1055667941`，训练损失仍在下降。
- epoch 129 左右的学习率约为 `5.2e-5`，原 150-epoch cosine scheduler 即将衰减到 0。
- 当前任务通过 `--resume best.pt` 恢复了旧 AdamW 和 scheduler 状态，因此不能开启独立的后期精调周期。
- 四样本验证成像中，`u_x` relative L2 为 `0.1385`，高于 `u_z` 的 `0.0848`。
- 训练无 NaN、Inf 或明显发散。主要问题是旧调度预算耗尽和 Marmousi 后期收敛缓慢，而不是优化器数值不稳定。

## 控制变量

短程筛选只改变优化器学习率。以下条件在所有候选中保持相同：

- 初始化权重：`artifacts/elastic_vti_pino_marmousi/checkpoints/best.pt`；
- 使用 `train.init_checkpoint` 加载模型权重，禁止使用 `--resume` 恢复旧优化器；
- 数据、split manifest、归一化统计、网络结构和损失权重不变；
- AdamW `weight_decay=1e-5`；
- 梯度裁剪 `grad_clip=0.5`；
- batch size 为 1；
- cosine scheduler 在每个候选内部重新开始；
- 固定 seed 和验证样本。

## 三候选短程筛选

候选初始学习率为：

1. `2e-5`：保守精调，最小化遗忘；
2. `5e-5`：接近当前最佳点附近的有效学习率，作为推荐中心候选；
3. `1e-4`：测试较强更新能否跳出当前平台。

每个候选运行 3 epochs，每 epoch 最多 64 个训练 batches，使用相同的最多 8 个验证 batches。每个候选写入独立目录，保存配置、日志、best/last checkpoint 和验证曲线。

候选按最低验证 relative L2 排序。若指标差异小于 `1e-4`，优先选择更低学习率。禁止用测试集、接收器图或人工视觉偏好选择候选。

## 接受门槛

候选必须同时满足：

- 验证 relative L2 严格低于 `0.1045`；
- 预测和梯度全部有限；
- 没有 OOM 或异常 loss 峰值；
- 使用与基线相同的 split 和归一化统计；
- 封存候选后进行四样本组件评估，`u_z` relative L2 相对基线恶化不超过 1%。

`u_x` 改善作为主要诊断报告，但不用于短程候选选择，以避免验证评估被重复拆分调参。若无候选达到门槛，则结果为“优化器精调未超过基线”，不替换当前最优权重。

## 正式精调

筛选胜出的候选从原始基线 `best.pt` 重新初始化，不能从三轮筛选后的候选 checkpoint 继续，以保证正式训练有完整且可解释的 scheduler 周期。

正式精调参数：

- 40 epochs；
- 使用筛选胜出的初始学习率；
- AdamW `weight_decay=1e-5`；
- `grad_clip=0.5`；
- cosine scheduler；
- early stopping patience 10；
- `early_stopping_min_delta=1e-4`；
- 每 epoch 保存 last，按验证 relative L2 保存 best。

正式运行写入新的 artifact 目录，不覆盖现有 Marmousi checkpoints、日志或评估图。

## 资源与执行顺序

1. 确认当前 Marmousi epoch 已保存 `last.pt` 后停止现有进程，释放约 9 GiB 显存。
2. 依次运行三个候选，避免并行训练造成显存竞争和不可比的时序干扰。
3. 汇总验证指标并封存选择结果。
4. 仅在候选通过接受门槛后提交正式 40-epoch 精调。
5. 记录停止的原任务 PID、候选命令、GPU 状态、checkpoint 哈希和结果目录。

## 非目标

- 不修改 modes、width 或层数；
- 不改变数据集或重新划分 split；
- 不同时调整 PDE、energy、receiver 或组件损失权重；
- 不承诺优化器调整必然超过模型容量上限；
- 不删除或覆盖当前 `best.pt`。

