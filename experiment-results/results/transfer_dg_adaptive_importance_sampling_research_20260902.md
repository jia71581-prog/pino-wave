# Transfer DG 自适应重要性采样调研与建议

日期：2026-09-02。目标：提高预训练记录、时窗、频率和 DG 单元采样的效率、覆盖充分性与实际预测收益。当前活动训练不修改；本报告用于下一轮 train-only 配对预注册。

## 1. 结论

建议新增 **Transfer DG-HCAIS**（Hierarchical Coverage-constrained Adaptive Importance Sampling）：

1. 分层配额保证 family、f0、时窗、频率、界面复杂度和边界类型不会被遗忘；
2. 使用误差、学习停滞、界面通量、高频误差和算子 leverage 构造难度分布；
3. 用 defensive mixture 保留固定探索概率；
4. 对非均匀抽样使用 inverse-probability 权重，保持目标风险估计一致；
5. 用 ESS、覆盖熵、Gram effective rank 和每 GPU-hour 精度收益监控采样是否真正有效。

纯 residual top-k、纯 loss-proportional sampling 或只增大困难样本权重都不建议。

## 2. 最新及关键文献

### 2.1 Residual-Christoffel Sampling（2026-07）

[Residual-Christoffel Sampling for Random Feature Collocation of Linear PDEs](https://arxiv.org/abs/2607.13382) 使用 operator-applied features 构造 residual-Christoffel 密度，配合 inverse-density weights 和 whitening，使采样后的 residual Gram 逼近参考 Gram；其确定性版本按 regularized log-determinant 增量逐行选点。

可迁移内容：在 Transfer DG local trace 数据中，不只选择大 flux error，还要选择能增加 trace/Jacobian Gram 秩的样本，避免反复采到高度相似的同一界面和频率。

限制：理论针对线性 PDE 的随机特征 collocation；用于深度神经算子时只能作为设计原则，不能直接继承其证明。

### 2.2 SARAS-PINN（2026-07，地震直接相关）

[SARAS-PINN](https://onlinelibrary.wiley.com/doi/10.1111/1365-2478.70230) 将结构感知初始化与 residual-gradient adaptive sampling 结合，在 layered、Marmousi 和 BP 速度模型上强调速度突变与盐体边界。其结果说明 residual-only 容易漏掉锐利结构，速度/解梯度应共同进入采样指标。

可迁移内容：将 `|grad log(c)|`、界面法向变化和 DG face-flux defect 作为独立于总 relative L2 的采样因子。

限制：目标是 eikonal traveltime PINN，不是全波场神经算子。

### 2.3 DAS-PINNs spacetime（2026-06）

[DAS-PINNs for spacetime domains](https://arxiv.org/abs/2606.06314) 用 normalizing flow 学习时空 residual-induced distribution，可跟踪移动和局部高残差区域。

可迁移内容：如果未来生成连续 source/velocity 参数，可用轻量密度模型提出新仿真实例。

限制：当前训练池只有离散的 240 条记录和有限时窗，直接训练 normalizing flow 的额外成本可能大于收益；第一版优先使用离散 alias sampling。

### 2.4 Self-adaptive weighting and sampling（2026-04）

[Self-adaptive weighting and sampling for PINNs](https://arxiv.org/abs/2511.05452) 将高梯度区域的自适应采样与 residual decay rate 权重结合；论文报告二者单独使用并不总是稳定，组合后更一致。

可迁移内容：采样分数不仅看当前 loss，还看 EMA loss 的下降速度。长期不下降的记录/频率比偶发大误差更值得增加概率。

### 2.5 Hessian-based provable refinement（2025）

[Provably accurate adaptive sampling](https://arxiv.org/abs/2504.00910) 用 residual Hessian 指导 quadrature refinement，减少高曲率区域的积分误差。

可迁移内容：对高频波场可用便宜的空间/时间二阶差分幅值作为 Hessian proxy，补足 residual 和一阶梯度未识别的窄波前。

### 2.6 Active learning for neural PDE solvers（2024）

[AL4PDE](https://arxiv.org/abs/2408.01536) 面向“数值求解器在环”的训练数据选择，比较 uncertainty 和 feature-based batch acquisition，并报告随机采样之外的平均及 worst-case 收益。

可迁移内容：在新增昂贵 LWC84 训练数据时使用模型委员会 disagreement + 参数空间 diversity，而不是继续随机扩充相似速度模型。

### 2.7 经典稳健基线

- [RAD/RAR-D](https://arxiv.org/abs/2207.10289)：使用 residual 幂与均匀常数项形成采样分布；实验表明中等自适应通常优于两个极端。
- [Causal R3](https://arxiv.org/abs/2207.02338)：retain-resample-release，并为时间 PDE 加入因果前沿，避免晚时刻困难点阻断从初边值向内部传播。
- [AAIS](https://arxiv.org/abs/2405.03433)：使用退火密度和 ESS 阈值处理多峰 residual 分布。
- [Importance sampling for PINNs](https://arxiv.org/abs/2104.12325)：给出按 loss-proportional density 加速收敛的理论和分片近似实现。

## 3. 当前采样的不足

现有 `ThreeWindowCache.balanced_indices` 已保证 `family x f0-bin` 轮转，这是正确的覆盖底座，但仍有四个不足：

1. 同一 stratum 内均匀抽样，不关注 Marmousi 界面密度、source-interface 几何或持续高误差记录；
2. slot 概率固定，不能根据各阶段真实学习进展调整；
3. high-frequency、interface 和 DG-flux defect 没有独立采样预算；
4. 改变抽样概率后若仍直接平均 loss，会改变目标风险，难以判断收益来自采样还是隐式 loss reweighting。

P1 的 slot-3 近零目标曾贡献约 99% 的梯度 proxy，说明只按原始相对误差或梯度大小做 importance sampling 会再次集中到病态样本。

## 4. Transfer DG-HCAIS 设计

### 4.1 目标单位

一个训练单位记为：

`i = (family, f0_bin, slot, source_record, element_type, frequency_bin)`。

全局时域预训练不使用 element/frequency 两项；local DtN 训练不使用 slot。所有自适应只作用于 fit split，holdout 始终完整、固定、均匀评分。

### 4.2 覆盖分布

定义基础分布 `q_cov`：

- family 和 f0-bin 固定轮转；
- 三个有效时窗均有概率下限，slot 3 继续排除；
- source depth/横向位置、界面密度、速度对比度按分位箱覆盖；
- local DtN 对 4 个相对频点、free-surface element、interface element 和 smooth element 设置最低配额；
- 每个 epoch 至少覆盖每个 source group 一次，不能让历史高损失记录永久占满。

### 4.3 难度分数

所有分量先在各自 stratum 内转换为 percentile rank，避免单位和幅值支配：

`d_i = 0.30*r(relative_L2) + 0.25*r(high_band) + 0.25*r(DG_flux_or_interface_H1) + 0.10*r(curvature_proxy) + 0.10*r(slow_decay)`。

- `relative_L2` 使用 onset-energy floor 后的稳健误差；
- `high_band` 使用目标频带能量 floor；
- `DG_flux/interface_H1` 针对散射界面；
- `curvature_proxy` 为波场二阶空间/时间差分；
- `slow_decay` 为样本 loss EMA 与前一周期相比的停滞程度。

分数只来自已访问的 train-fit 标签/残差；未访问样本使用 stratum 中位数和探索 bonus，不读取 validation/test。

### 4.4 Leverage/充分性分数

对 local DtN trace 或最终 decoder correction basis，维护低维特征 `phi_i`：

- pressure trace DCT/plane-wave coefficients；
- operator residual Jacobian 对最后一层/低维系数的投影；
- velocity-interface 和 frequency embedding。

用 regularized leverage 或 log-det 增量：

`l_i = phi_i^T (G + lambda I)^(-1) phi_i`

衡量样本能否增加已见子空间的独立方向。只看 residual 会重复选同类困难样本，leverage 用于保证信息充分性。

### 4.5 防御混合采样分布

第一版建议：

`q_i = 0.40*q_cov_i + 0.40*q_difficulty_i + 0.20*q_leverage_i`。

其中 `q_difficulty proportional to (d_i + epsilon)^tau`，`tau` 从 0 在前 20% epoch 线性退火到 1。固定 40% coverage 意味着 importance weight 有上界，避免尾部爆炸。

如果 batch 的有效样本量：

`ESS = (sum w_i)^2 / sum(w_i^2)`

低于 `0.5 * batch_size`，自动提高 `q_cov` 比例，直到 ESS 恢复。第一版不使用 weight clipping，以免引入额外偏差。

### 4.6 重要性校正

若目标仍是原始均匀/分层风险 `p_i`，每个训练样本必须乘：

`w_i = p_i / q_i`。

对固定 stratum 配额，使用条件权重 `p(i|stratum)/q(i|stratum)`。这样改变的是估计方差和样本利用率，而不是悄悄改变主目标。

自适应 weighting 只用于不同物理 loss component 的收敛平衡，必须与 sampling weight 分开记录。

## 5. 三层落地方案

### 5.1 当前时域预训练

- 保留 family x f0 轮转；
- 在 stratum 内按记录难度与 source/medium diversity 采样；
- slot 先保留当前 0.60/0.25/0.15 coverage，最多只让自适应项调整其中 40%；
- 每个 epoch 更新记录 EMA，不额外完整渲染候选池；
- 每 2 个 epoch 做一次 train-fit score refresh。

### 5.2 Local DtN / DG elements

- batch 先按 family、frequency 和 element type 分层；
- 40% 采高 flux-error/高界面梯度样本；
- 20% 按 trace-feature leverage 或 greedy log-det 采样；
- 40% 保持覆盖；
- source-group holdout 永不参与自适应。

由于 local dataset 只有 6,912 条，全部样本打分成本很低，这里最适合先验证 HCAIS。

### 5.3 新 LWC84 数据生成

对未标注 velocity/source pool 使用：

`acquisition = ensemble_disagreement + parent_DG_defect + high_frequency_risk + diversity`。

用 k-center/log-det 批量去冗余后才调用昂贵 teacher。PGD adversarial mining 只允许在物理参数范围、CFL、最小 ppw 和速度平滑/界面约束内进行。

## 6. 充分性与有效性审计

必须同时报告：

- **覆盖**：每个 stratum/分位箱计数、最小概率、覆盖熵、最长未访问 epoch；
- **有效样本量**：ESS 均值、P5、最小值和 importance weight 最大值；
- **信息秩**：trace/Jacobian Gram effective rank、log-det 和条件数；
- **效率**：达到同一 holdout 精度所需 updates、samples、GPU-minutes；
- **效果**：aggregate、family、时窗、high-band、interface、free-surface、nonworse；
- **开销**：score refresh 和 sampler 本身占总训练时间的比例。

若只降低训练 loss、只改善被高频抽到的子集、ESS 崩溃、覆盖缺失或采样开销超过收益，均判定失败。

## 7. 最小可归因实验

在 local DtN 数据上先做相同 seed、相同模型、相同 batch 数的配对：

1. control：当前 source-group-balanced uniform sampler；
2. HCAIS：40% coverage + 40% difficulty + 20% leverage，并带 inverse-probability weights。

主判据仍按用户要求：完整固定 holdout 的 mean robust relative flux error 严格改善，不设最低百分比。附加要求是 family 不缺失、ESS P5 不低于 batch 的 50%、采样开销不超过总训练时间 10%。通过后再移植到全局时域预训练。

## 8. 不建议直接采用

- residual top-k/RAR-G 作为唯一策略；
- 无 coverage floor 的 loss-proportional sampling；
- 改采样概率但不做 importance correction；
- 用 validation residual 更新采样器；
- 对 slot-3 近零记录使用原始 relative loss 评分；
- 在 240 条离散记录上先训练复杂 normalizing flow；
- 用全网络逐样本梯度范数，成本过高；第一版只用最后一层或低维 basis 的 Jacobian proxy。

