# Continuous Wave Operator 设计规范

日期：2026-07-17
项目：`/home/jiayh/Data/FNO-Acoustic-Wave-Simulation`

## 1. 目标与范围

在仓库根目录新增独立的 `continuous_wave_operator/` 算法包，面向二维常密度变速声波方程数据建立连续时空、介质与震源解耦的神经算子。模型对任意物理坐标 `(x,z,t)` 输出压力，可在同一介质编码上复用多个震源，并为以后基于接收器数据的 FWI 保持关于速度场、坐标和时间的可微性。

本轮包含模型、数据加载、训练、验证、checkpoint、推理接口、配置、测试和正式训练。不实现 FWI 优化器、反演正则化或模型更新流程。

## 2. 数据合同与前置修复

正式数据源为：

`/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5`

监控日志 `artifacts/production_launch_logs/lwc84_401to201_batch128_monitor.log` 仅用于运行审计，不是训练数据。

数据合同：

- `velocity_mps`: `[N,201,201] float32`
- `wavefield`: `[N,401,201,201] float32`，轴顺序 `NTZX`
- `source_map`: `[N,201,201] float32`
- `source_wavelet`: `[N,401] float32`
- `time_s`: `[401] float64`，覆盖 `0–1 s`
- `x_m,z_m`: `[201] float64`，覆盖 `0–2000 m`
- 震源标量：`source_x_m,source_z_m,source_f0_hz,source_t0_s,source_amplitude`
- 标签与溯源：`split,medium_type,sample_id,group_id,seed`

当前已生成 3968/4003 个样本；train 2800 和 validation 600 完整，test_id 缺 32 个，OOD 缺 3 个。生产中断根因是固定分层/OOD manifest 参数不含 `medium_parameters.seed`，而 `_production_velocity` 无条件访问该键。实施必须先通过回归测试修复固定分层参数解析，使用 `--resume` 补齐剩余样本，再构建并严格校验统一 VDS。不得删除或重算已完成分片。

## 3. 公共模型接口

核心张量：

```text
velocity_mps  [B,1,201,201]
source_map    [B,S,1,201,201]
source_params [B,S,5]        # x_s,z_s,f0,t0,amplitude
query_coords  [B,S,Q,3]      # x,z,t，使用物理单位
pressure      [B,S,Q]
```

公开接口：

```python
medium = model.encode_medium(velocity_mps)
sources = model.encode_sources(medium, source_map, source_params)
pressure = model.query(medium, sources, query_coords)
pressure = model(velocity_mps, source_map, source_params, query_coords)
```

`medium` 缓存必须可被同一介质的多个震源和多个查询块复用。`query` 支持将大量查询拆成 4096–32768 点的块，同时保持数值结果一致。

## 4. 模型架构

### 4.1 介质编码器

原始速度按参考速度 `c_ref` 形成三个无量纲通道：`c/c_ref`、`(c/c_ref)^2`、`c_ref/c`。编码器采用四级局部残差卷积与二维谱卷积混合结构，空间尺度为 `201→101→51→26`。各尺度特征供连续坐标双线性采样；最高层再自适应压缩为约 `12×12` 个全局介质 tokens。

介质编码器不得依赖震源信息，因此同一速度模型的多炮计算只执行一次介质编码。

### 4.2 震源编码器

`source_map` 在各介质尺度上做质量守恒的降采样和加权池化，取得震源邻域介质特征。该特征与 `(x_s,z_s,f0,t0)` 的 Fourier 特征融合形成 `source_latent [B,S,C]`。

震源幅值不作为任意隐藏变量学习；最终压力显式乘以 `source_amplitude`，从结构上保持线性声学方程对源幅值的线性关系。

### 4.3 连续查询解码器

查询解码器融合：

- `(x,z,t)` 的连续 Fourier features；
- 查询位置在各尺度介质图上的局部采样特征；
- 压缩介质 tokens 的交叉注意力；
- source latent 产生的 FiLM 条件；
- 平滑残差 MLP。

模型输出标量压力。交叉注意力使用压缩 tokens 并支持查询分块，避免形成完整 `401×201×201` GPU 张量。

### 4.4 物理归纳偏置

- 输入物理坐标内部按 `Lx=2000 m,Lz=2000 m,T=1 s` 无量纲化。
- 输出时间门保证 `p(t=0)=0` 且 `p_t(t=0)=0`。
- 顶部门保证 `p(z=0,t)=0`，匹配压力自由表面。
- 左、右、下不施加零边界，因为标签来自三侧 CPML。
- 模型路径保持对速度场、查询坐标和时间的 PyTorch 自动微分能力。

## 5. 文件边界

```text
continuous_wave_operator/
├── __init__.py
├── config.py
├── coordinates.py
├── spectral_layers.py
├── medium_encoder.py
├── source_encoder.py
├── query_decoder.py
├── model.py
├── data/
│   ├── dataset.py
│   ├── query_sampling.py
│   └── grouped_shots.py
├── training/
│   ├── losses.py
│   ├── adaptive_sampling.py
│   ├── checkpoint.py
│   ├── trainer.py
│   └── validation.py
├── configs/
│   ├── smoke.yaml
│   └── production.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── query_receivers.py
├── tests/
└── README.md
```

实现只依赖仓库现有 Python/PyTorch/HDF5 栈，不引入外部神经算子框架。各模块通过明确数据类或张量合同通信，不直接依赖现有旧 FNO 的内部状态。

## 6. 查询数据加载

训练不把整个时空波场搬到 GPU。加载器按 HDF5 的逐时间帧 chunk 选择少量时间帧，再从所选帧抽取空间查询点。每个训练样本读取一次速度、震源和坐标元数据。

初始查询混合：

- 50% 全域均匀点；
- 25% 波场高能量或残差热点；
- 25% 近地表及随机接收器线。

时间采样分层覆盖早期直达波、中期反射/散射和晚期衰减。同一 `group_id` 中介质完全一致的样本可组成多炮 batch；加载器必须先验证速度哈希或逐值一致性，不能仅凭字符串假定可复用介质。

## 7. 三级自适应重要性采样

### 7.1 样本级

在四类介质基础平衡的前提下，为每个 train sample 维护监督误差与 PDE residual 的指数移动平均，提升困难样本的后续概率。

### 7.2 时间级

将 `0–1 s` 划分为固定时间区间，为每个区间维护 residual EMA，动态分配查询帧，同时保证早、中、晚期均有非零最小概率。

### 7.3 空间级

每个样本维护低分辨率 residual 热图。查询点由均匀分布、残差热点和接收器区域混合产生，并映射为连续物理坐标；不保存完整时空 residual 张量。

采样混合初值：

```text
uniform_fraction        = 0.25
family_balanced_fraction= 0.25
residual_fraction       = 0.50
ema_momentum             = 0.10–0.25
residual_power           = warmup 后由 0 增至 1
```

训练损失使用受裁剪的逆概率校正，防止 AIS 改变经验风险或因极小概率产生梯度爆炸。validation/test/OOD 使用固定确定性查询。AIS 概率、EMA、随机数状态及更新步数全部保存到 checkpoint，恢复训练时严格校验。

## 8. 损失与训练阶段

总目标：

```text
L = L_data + lambda_t L_trace + lambda_f L_spectral + lambda_p L_PDE
```

- `L_data`：归一化 Charbonnier 与 MSE 混合查询损失；
- `L_trace`：固定接收器完整时间波形损失；
- `L_spectral`：接收波形频谱幅值和相位损失；
- `L_PDE`：远离离散点源及保存域侧边界的声波方程 residual。

训练分两阶段：

1. 监督预训练，学习全介质、全时间传播映射并启动 AIS warmup；
2. 物理微调，逐步增加 PDE 与频谱权重，继续介质族平衡 AIS。

归一化统计只从 train split 流式计算。validation selection metric 固定，不得用 test/OOD 调参。

## 9. 错误处理与可复现性

- 数据文件、schema、轴顺序、坐标、split 数量或样本校验失败时硬退出。
- 非正速度、非有限输入/输出/梯度和非法查询范围硬退出并记录 sample ID。
- checkpoint 记录配置、Git commit、数据 manifest/VDS 源分片哈希、模型/优化器/调度器、归一化、AIS 和随机数状态。
- resume 必须校验模型配置、数据绑定和 AIS 结构；不允许静默部分加载。
- 训练输出写入新的 `artifacts/continuous_wave_operator/`，不覆盖现有 FNO/PINO 实验。

## 10. 验证与验收

实施采用测试驱动开发，至少覆盖：

- `B,S,Q` 形状、单炮/多炮、查询分块一致性；
- 同一 medium cache 多炮复用且不重复编码；
- source amplitude 线性；
- `t=0` 初始条件与 `z=0` 自由表面硬约束；
- 对速度和 `(x,z,t)` 的有限、非零梯度；
- VDS/shard schema、split 和 NTZX 轴读取；
- group 多炮复用前的速度一致性检查；
- 三级 AIS 的概率下限、确定性、残差偏置与逆概率权重；
- checkpoint/resume 完整恢复 AIS 与 RNG；
- CPU 小模型 smoke train；
- 数据完成后 CUDA 单步与短训练 smoke，不与数据生成并发争抢 GPU。

正式训练启动前必须满足：4003/4003 样本完成、统一 VDS 严格校验通过、相关测试通过、GPU 空闲且磁盘预算足够。正式指标只能来自固定 validation/test/OOD 协议。
