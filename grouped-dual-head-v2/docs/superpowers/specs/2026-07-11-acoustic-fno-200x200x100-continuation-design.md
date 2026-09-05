# Acoustic FNO 200x200x100 自适应续训设计

## 目标

从当前声学算子验证集最优权重继续训练，将网络实际输入提高到完整空间网格 `200 x 200` 和前 `100` 个保存时间步。正式任务只能在 GPU smoke 通过后用 `nohup` 提交。

## 初始化与数据契约

- 初始化检查点：`artifacts/hybrid_drp_pino/fno_400x54_spatial_residual_importance/checkpoints/best.pt`。
- 使用检查点内保存的训练/验证/测试 split manifest，禁止重新划分数据。
- 复用检查点归一化统计，避免分辨率变化被混入归一化漂移。
- 原始数据仍为 `/home/jiayh/Data/data/pino.hdf5`。
- 采用抗混叠双线性空间重采样到 `200 x 200`。
- 时间轴严格取数据契约中的前 100 个保存时刻，不重新等间隔构造时间坐标。

## 模型与优化

- 保持原 Acoustic FNO3D 结构：4 层、宽度 32、三轴 Fourier modes 均为 16。
- `batch_size=1`；这里表示每个优化 step 使用一个样本，不是只训练一个 batch。
- 启用 activation checkpointing，冻结 BatchNorm 运行统计。
- 初始学习率 `1e-6`，AdamW，weight decay 为 0，cosine scheduler。
- 梯度裁剪上限 `0.5`，禁止 AMP 改变非 2 次幂网格 FFT 的数值行为。
- 第一阶段只使用受监督 relative L2 和小权重 MSE，不同时引入新的 PDE/频散损失。

## 自适应重要性采样

- 样本级重要性采样：每 epoch 256 个训练样本，25% 保持均匀抽样。
- 空间重要性采样：每个样本选 8192/40000 个空间像素，25% 保持均匀抽样。
- 使用残差 EMA 更新采样权重，并对空间抽样损失执行无偏重加权。
- 第一个 epoch 没有历史残差时退化为均匀采样。

## 稳定性 Smoke

正式训练前运行独立 smoke 配置：

- 使用 `200 x 200 x 100`、`batch_size=1` 和正式模型权重；
- 至少完成 2 个前向/反向优化 step；
- 完成小规模验证；
- 成功写入并重新读取 checkpoint；
- 记录峰值 allocated/reserved GPU memory；
- loss、梯度、预测和验证指标全部为有限值；
- 峰值 reserved memory 不超过可用 RTX 3090 显存并保留安全余量。

任何一项失败时不得提交正式任务。显存不足时优先减少 `pixels_per_sample` 或暂停其他 GPU 任务，不降低 `200 x 200 x 100` 主输入网格。

## 正式训练与产物

Smoke 通过后使用 `nohup` 启动正式训练。正式运行使用独立目录，保存：

- 解析后的配置和初始化检查点 SHA256；
- best/last checkpoints；
- 每 epoch 训练损失、验证 relative L2、MSE 和学习率；
- 样本级与空间级残差 EMA；
- 显存峰值、有限值比例和失败原因；
- PID、启动命令和 nohup 日志路径。

正式任务不得覆盖现有 `400 x 400 x 54` 检查点或评估结果。

## 验收

- 实际模型输入形状为 `[1, 200, 200, 100, 3]`，输出为 `[1, 200, 200, 100]`。
- 初始化权重哈希与指定 `best.pt` 一致。
- Smoke 训练和验证均无 OOM、NaN 或 Inf。
- 正式 nohup 进程启动后至少成功进入首个训练 step，并将日志写入新运行目录。

