# Transfer DG：最新神经算子与有限元/DG 结合方法调研

日期：2026-09-02。范围：截至本日可检索的原始论文与预印本。开发协议保持 `validation_opened=false`、`test_id_opened=false`。

## 结论

可以结合，而且建议把 Transfer DG 的完整结构升级为：

`全局神经算子先验 + 可复用局部神经单元 + 能量稳定的 DG/HDG trace-to-flux 组装 + 低维在线 trace 校正`。

当前运行的保守界面 flux block 是这个方向的最小原型。若它通过 train-only 配对门槛，下一版应升级为局部算子单元；若失败，也不代表神经算子与 DG 不可行，只说明“单个低秩 latent jump block”容量不足。

## 1. 最相关的最新工作

### 1.1 NN-MsHDG（2026-08-26）

[A Neural-network-based multiscale Hybridizable Discontinuous Galerkin method](https://arxiv.org/abs/2608.25850) 保留 MsHDG 的局部到全局结构，让网络预测每个粗块的离散 Dirichlet-to-Neumann（DtN）矩阵和 source vector，再在 skeleton 上组装。低/中对比度实验报告约 5--16 倍在线加速。

最值得迁移到 Transfer DG 的机制：

- 网络学习局部 `trace -> numerical flux`，而不是直接学习完整全局场；
- source-induced 与 trace-induced 响应分开；
- 局部单元可批量复用，全局物理连接仍由 HDG skeleton 保证；
- 用 operator-action loss 检查 DtN 在相关 trace 方向上的作用；
- 保留物理 nullspace。

论文同时给出重要警告：高对比度下固定低阶 polynomial trace space 误差较大，局部网络误差会被全局组装放大。因此我们不能直接照搬固定 P1 trace，需要使用速度与频率条件化的 spectral trace basis。

### 1.2 Convex Neural Energy Elements（2026-08-03）

[Convex Neural Energy Elements](https://arxiv.org/abs/2608.02036) 指出：局部场预测即使只有约 1% 回归误差，诱导出的组装 Hessian 仍可能不定，最终 Newton 解出现 247% 误差。该方法改为学习标量能量，并从结构上保证边界自由度上的凸性、刚度正半定和正确 nullspace。

对声波问题的直接启示不是照搬静态凸能量，而是参数化可证明稳定的局部离散算子：

- mass matrix 为 SPD；
- stiffness/储能部分为 PSD；
- 中心通量部分满足离散反对称/能量守恒结构；
- upwind/CPML 部分只引入非负耗散；
- 正则项不能破坏声学常数/刚体式 nullspace。

因此 Transfer DG 应学习稳定的局部能量、通量或算子因子，不应只回归 pressure field 后再数值求导。

### 1.3 NOEM（2025 预印本，2026 Nature Computational Science）

[Neural-operator element method](https://arxiv.org/abs/2506.18427) 将神经算子做成可复用 neural-operator element，用它替代需要大量细单元的子域，再与普通有限元通过变分框架组装。工作覆盖多尺度、复杂几何与不连续系数。

对本项目的价值：可把 Marmousi/强界面区域作为 neural elements，均匀区域保留便宜的解析或粗传播；不同区域不必使用同等网络容量。

### 1.4 DG-FEONet 与 Sparse FEONet（2026-01）

[DG-FEONet](https://arxiv.org/abs/2601.03668) 让网络预测逐单元 DG 系数，并最小化 SIPG 弱形式残差，面向不连续系数和非光滑解。[Sparse FEONet](https://arxiv.org/abs/2601.00672) 利用有限元局部稀疏性降低 FEONet 的大网格成本。

可迁移机制：

- element-wise coefficients；
- volume residual 与 interface jump/penalty 分开；
- 稀疏邻接只连接共享面的单元；
- loss 和网络复杂度随局部 stencil，而不是全域 dense matrix 扩张。

限制：目前主要证据来自椭圆/对流扩散问题，不能直接声称适用于带 CPML 的高频时域声波。

### 1.5 FEONet/FOL 与时空弱形式

- [FEONet](https://arxiv.org/abs/2308.04690) 直接预测有限元系数，并使用变分残差训练，说明理论基函数有助于边界层和尖锐结构。
- [Finite Operator Learning](https://arxiv.org/abs/2407.04157) 将离散弱形式、边界条件和 Sobolev 目标写入 operator loss。
- [Finite-element physics-informed operator learning for spatiotemporal PDEs](https://arxiv.org/abs/2405.12465) 用有限元弱形式和离散时间步训练时序算子。
- [Element learning](https://arxiv.org/abs/2308.02467) 学习 element-level `in2out` 与 `in2sol` 两个算子，与 HDG 的局部消元非常接近。

这些工作支持“局部单元算子 + 组装”，但其中常用的 implicit Euler 对波动问题会引入相位耗散，不能直接用于我们的高频声波。

### 1.6 最新声学直接证据（谨慎采用）

[FNO + finite-element guided graph refinement for ocean acoustics](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7265840) 是 2026-08-11 的 SSRN 预印本：FNO 先给全局 Helmholtz 场，随后按 FE 矩阵稀疏图做局部修正，矩阵条目作为 edge features，并按频率融合。它与当前 Transfer DG 最接近，但尚属早期、未充分同行评议证据。

可吸收的部分是 FE/DG 矩阵稀疏结构和 edge coefficient；不直接采用其结果数字作为我们时域问题的性能承诺。

## 2. 建议的 Transfer DG v2 架构

### 2.1 域分块

将 201x201 网格划分为约 12--25 格宽的 coarse elements。速度梯度低的区域使用大单元，界面、薄层和低 points-per-wavelength 区域细分。分块规则只由 velocity、f0 和网格决定，不读取目标波场。

### 2.2 局部神经算子单元

每个 element 共享一个条件化 local operator，输入为：

- 局部 `c(x,z)`、slowness、界面法向和反射系数；
- f0、t0、局部 points-per-wavelength；
- 四条边上的 incoming pressure/normal-velocity trace；
- source-induced local forcing；
- 自由表面或 CPML element type。

输出不要只给 interior pressure，而应给：

1. `trace -> outgoing numerical flux` 的 DtN/operator action；
2. source-induced flux vector；
3. 可选 interior pressure reconstruction；
4. 局部 energy/stability factors。

所有 element 在 GPU 上批量推理，参数跨位置共享。

### 2.3 系数自适应 trace basis

固定 P1/P2 trace 对高对比度和高频不够。建议离线学习条件化 trace basis：

- 低频：Legendre/P1 基；
- 高频：局部 Fourier/plane-wave/Trefftz 基；
- 界面：由阻抗和法向条件化的 reflected/transmitted basis；
- basis 数由 f0 和局部最小 points-per-wavelength 决定。

训练时同时约束 basis 正交性、DtN action、nullspace 和跨分辨率稳定性。

### 2.4 全局组装

对于当前时域任务，不建议每帧求解大型全局 FEM 系统。更合适的是显式、能量稳定的 DG skeleton 更新：

- element interior 由 local neural operator 更新；
- 共享面通过阻抗加权 acoustic Riemann/upwind flux 交换；
- 顶面使用 pressure-release flux，并保持硬 `p=0`；
- 左/右/底 CPML 只有在同时表示 CPML auxiliary states 后才进入可训练组装；
- 组装算子按守恒/耗散结构参数化，防止局部小误差被全局放大。

另一条可选路线是在少量频率上学习 Helmholtz local DtN，然后并行组装并重建时域。它对高频散射更自然，但需要单独验证频域到时域的相位和运行时间。

## 3. 与当前正在训练版本的关系

当前 `DGInterfaceFluxResidual2d` 已有：

- velocity reflection edge feature；
- element-local latent jump；
- equal/opposite conservative scatter；
- CPML 三侧屏蔽；
- zero-gated P1b transfer。

缺少：

- 显式 element 分块和 trace DOFs；
- source/trace 分解；
- 学习的 local DtN action；
- energy-stable matrix factorization；
- coefficient-adapted high-frequency trace basis；
- pressure/normal-velocity 或 CPML auxiliary physical states。

所以当前版本适合作为最便宜的机制筛查，不应视为最终 Transfer DG。

## 4. 推荐实验顺序

1. 完成当前 DG-flux 与 control 的两种子配对门槛。
2. 构建 train-only local-element 数据：局部 velocity/source、边界 traces、LWC84 一致的 normal flux 与 interior response。
3. 先训练 `trace -> flux action`，而不是预测 dense DtN matrix；用随机 trace、真实波前 trace 和高频 plane-wave trace共同训练。
4. 比较三种局部目标：field-only、flux-action、energy/stability-factor。根据最新能量单元结果，field-only 只作为反例。
5. 在 2x2、4x4、完整 201x201 多单元组装上检查误差放大和离散能量。
6. 通过后再将 onset 观测用于低维 skeleton/basis 系数的在线凸校正。

主门槛继续按用户要求：同协议 aggregate relative L2 只要严格改善即可，不设置最低百分比；同时必须报告 family、界面带、高频带、nonworse 和端到端运行时间。

## 5. 明确不建议的组合

- 不恢复已经失败的连续 Q1-FE weak loss；其 12 条训练面板总体恶化约 1.82%。
- 不把完整 FEM/DG 正演塞入在线微调，否则很可能失去 10x 速度目标。
- 不学习 field 后无约束组装；最新结果表明局部场误差小不代表组装稳定。
- 不用固定低阶 trace basis 覆盖所有 f0/速度对比度。
- 不用 implicit Euler 作为声波预训练的时间弱形式。
- 不在缺少 CPML auxiliary states 时伪造 CPML DG residual。

