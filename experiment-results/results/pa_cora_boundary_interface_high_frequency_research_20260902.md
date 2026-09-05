# PA-CORA 边界、界面散射与高频表达补充研究（2026-09-02）

## 结论

自由表面边界违规已用参数无关的硬投影修复，但它只解释当前误差的一小部分。下一版预训练模型应保留硬边界，并把主要容量转向“散射残差分解 + 多尺度频率支路 + 界面局部支路 + 离散弱形式物理约束”。只给现有网络增加物理输入通道不够：已有 physics-cond 两种子结果（0.11891、0.11921）均弱于 spectral 两种子（0.11421、0.11512）。

本轮分析只使用 train split 的 group-disjoint holdout；`validation_opened=false`，`test_id_opened=false`。

## 1. 已完成的边界修复

实现：

- 输出硬投影：对每一帧严格执行 `p[..., z=0, :]=0`。
- 与数据生成器一致的 halo：顶面八阶奇延拓，左、右、底三侧零 halo。
- 参数无关包装器可直接包裹已有检查点，不改变其内部参数。
- 三项单元测试覆盖误差非增性、halo 规则以及普通/anchored 前向输出。

在 P1b seed 372、epoch 8 检查点上的三窗口 train-only 审计结果：

| 指标 | 原始模型 | 硬边界模型 | 结果 |
|---|---:|---:|---:|
| 三窗口平均 relative L2 | 0.28012002 | 0.28009265 | 严格改善 0.00977% |
| onset relative L2 | 0.11331504 | 0.11325878 | 严格改善 0.04965% |
| 自由表面最大绝对违规 | 0.17855172 | 0 | 完全修复 |
| 自由表面 RMS（记录均值） | 0.00263447 | 0 | 完全修复 |
| 内部区域预测最大变化 | - | 0 | 内部逐点不变 |
| 改善/持平/变差记录数 | - | 72/0/0 | 全记录非劣 |

按窗口分解：

| 窗口 | 原始 relative L2 | 投影后 | 相对改善 |
|---|---:|---:|---:|
| slot 0 / onset | 0.11331504 | 0.11325878 | 0.04965% |
| slot 1 | 0.28620568 | 0.28619150 | 0.00496% |
| slot 2 | 0.44083933 | 0.44082766 | 0.00265% |

因此，“未严格满足边界”是确定存在的问题，但不是 0.11--0.44 误差的主因。投影应永久保留，同时不应把后续预算继续集中在顶面一个网格行。

## 2. 当前结构为什么难以学习界面散射与高频

当前 P1b 的可训练修正器是 width 64、depth 4、modes 24 的递归 factorized complex FNO。每个谱层分别沿 x 和 z 做一维变换，再相加；`coupled_axes=false`、`coupled_2d_rank=0`。局部分支只有 3x3 depthwise 卷积，解码器的有效局部感受野也较小。

这带来三个直接瓶颈：

1. 截断到 24 个轴向模态会优先拟合低频，难以恢复窄波前、高频尾部和相位细节。
2. 独立 x/z 频谱缺少显式二维波数耦合，对倾斜界面、曲面界面和多次散射的表达效率不足。
3. Fourier 全局基面对速度突变存在 Gibbs 型困难，而现有局部分支不足以专门修复界面附近的高梯度残差。

数据诊断与此吻合：V9 onset 的 interface-top10 误差约 0.134、non-interface 约 0.047，Marmousi interface 约 0.201；后续窗口 high-band 误差约 0.707/0.824。小范围时间平移 oracle 只能解释少量误差，因此主要不是统一时间偏移，而是散射振幅、相位和局部高频结构未被表达。

## 3. 文献证据与可迁移机制

- [MscaleFNO](https://arxiv.org/abs/2412.20183) 用多个输入与坐标缩放不同的并行 FNO 降低谱偏置，并在高频 Helmholtz 散射中优于参数量相近的普通 FNO。它支持为当前网络增加受 f0/points-per-wavelength 控制的多尺度谱支路，而不是只扩大单个 `modes`。
- [Learned frequency-domain scattered wavefield solutions](https://arxiv.org/abs/2405.01272) 把源位置和频率编码进背景波场，并学习 scattered wavefield；其现实速度模型实验直接支持“背景/直达场 + 散射残差”的目标分解。
- [U-FNO](https://arxiv.org/abs/2109.03697) 在 Fourier 层旁加入 U-Net 路径，以同时保留全局依赖和局部非线性结构；这与当前误差集中在界面和复杂 Marmousi 局部区域的现象相符。
- [Multiwavelet operator](https://arxiv.org/abs/2109.13459) 在多尺度局部基上学习算子核，可在低分辨率训练后迁移到更高分辨率；[Wavelet Neural Operator](https://arxiv.org/abs/2205.02191) 也强调小波的时空/频率局部化。两者支持用固定 Haar/多小波高通特征为界面支路提供细节，而不让全部通道承担高频。
- [OPNO](https://arxiv.org/abs/2206.12698) 针对非周期边界构造满足 Dirichlet/Neumann/Robin 条件的谱算子，并证明严格边界满足。当前实现先采用更简单且零风险的硬投影；若 matched-halo 仍不足，再考虑 z 方向正弦/Chebyshev 基、x 方向 Fourier 的混合谱层。
- [Deep Neural Helmholtz Operators](https://arxiv.org/abs/2311.09608) 使用 U 形神经算子处理 2D/3D 弹性波，并报告频率域并行和表面图算子带来的优势，支持将 U 形局部路径和频带并行作为离线高成本预训练方案。
- [GreenONet](https://arxiv.org/abs/2307.13902) 将波动方程 Green 函数解结构加入算子，支持使用解析直达场/背景响应作为源编码，而不是只给网络一个 Gaussian source map。
- [WHNO](https://arxiv.org/abs/2511.07347) 的结果表明 Walsh-Hadamard 基和 Fourier 基对不连续区域与平滑区域具有互补性。该工作尚未在声学时域散射上验证，因此只列为后续探索项，不作为第一轮主干。

## 4. 建议集成为新方法：BI-SM PA-CORA

暂定全称为 **Boundary-consistent Interface-scattering Multiscale PA-CORA**。它不是把多个现有模型简单拼接，而是围绕当前任务的误差分解形成一个统一残差算子：

### 4.1 输入物理编码

保留 warp_r1 的快速背景波场，并新增以下部署合法特征：

- `log(c)`、slowness-squared、`grad log(c)`、界面梯度幅值与法向；
- x/z 邻接速度反射代理 `(c_next-c)/(c_next+c)`，多尺度 5x5/17x17 速度对比；
- f0、t0、解析走时相位的 sin/cos、局部 points-per-wavelength；
- 顶面 signed-distance/free-surface mask；
- CPML 的真实 `sigma_x/z`、`kappa_x/z`、`alpha_x/z` 和阻尼距离，而不只使用边界距离及一个最大速度代理；
- 基于源点局部速度的解析直达/背景场，作为 source-frequency-location 的联合编码。

已有 20 通道 physics-cond 单独使用时不如 spectral arm，说明这些特征必须通过独立物理编码器、归一化和门控注入，不能直接拼接后交给同一 1x1 投影。

### 4.2 三支路残差算子

1. **Global recurrent branch**：继承当前共享时间步递归和 warp anchor，负责低/中频长程传播。
2. **Multiscale spectral branch**：采用 1/2/4 三个尺度的并行 FNO；坐标、速度特征和相位特征同步缩放，由 f0 与最小 points-per-wavelength 产生门控。至少加入一个低秩完整 2D rFFT 分支以表达斜向波数耦合。
3. **Interface-local branch**：以软界面掩码为门控的轻量 U-Net，并输入固定 Haar/多小波高通系数，专门预测反射、透射与绕射残差。

三支路只预测相对于 warp/background 的修正，最终输出统一为：

`p_hat = P_free_surface(p_warp + delta_global + delta_multiscale + delta_interface)`。

所有新增支路零门控初始化，保证从当前 P1b 检查点迁移时初始输出等于“P1b + 硬边界”，可以逐项归因。

### 4.3 损失函数

建议使用记录内归一化的组合目标：

`L = L_rel + lambda_f L_band + lambda_i L_interface_H1 + lambda_s L_scatter + lambda_w L_FE_weak + lambda_t L_multiphase`。

- `L_rel`：每记录 relative L2，保留 onset-energy denominator floor，避免近零晚窗主导梯度。
- `L_band`：按目标频带能量归一化的低/中/高频复谱误差，高频权重由 f0/ppw 平滑调节。
- `L_interface_H1`：在速度梯度膨胀带内计算压力与空间梯度误差；每记录重新归一化，防止 Marmousi 像素数支配训练。
- `L_scatter`：监督背景场之外的散射残差，并同时保留总场误差，避免只拟合平滑直达波。
- `L_FE_weak`：在非 CPML 内域使用 Q1 单元测试函数计算 `p_tt/c^2 - Laplacian(p) - q` 的离散弱残差。离线数据增加细时间步 triplet，避免用稀疏保存帧近似高频二阶时间导数。弱形式天然跨越速度界面，且不需要对不连续系数做强形式求导。
- 自由表面不再使用软 penalty，因为硬投影已经精确满足。

## 5. 训练数据应补充的分布

保持现有 uniform/layered/Marmousi，同时为预训练增加可控散射 curriculum：

- 速度对比度、界面曲率/倾角、薄层厚度、断层位移、多尺度异常体；
- 近地表反射、不同入射角到达 CPML、源到界面/顶面的相对距离；
- f0 与最小 points-per-wavelength 联合分层，提升高频和低 ppw 样本比例；
- 按 `family x f0 x interface-density x contrast x incidence-angle` 平衡采样；
- 保存细时间步 triplet、真实 CPML profiles、背景直达场和散射场监督。

离线生成与预训练代价不设上限，但部署输入不能依赖未来真值或在线有限元求解。

## 6. 最小可归因实验顺序

1. **A0**：P1b seed 372 + 已通过的硬边界投影，作为新基线。
2. **A1**：只加入 matched odd-top/three-side halo，检查边界邻域传播而非单行投影。
3. **A2**：A1 + 三尺度频谱 + 低秩完整 2D 波数耦合。
4. **A3**：A2 + 界面门控 U-Net/小波支路。
5. **A4**：A3 + 背景/散射分解及新物理编码器。
6. **A5**：A4 + FE weak loss 与细时间步 triplet。

每个阶段采用两主种子、两辅助种子并行占用四卡；主判据仍按用户要求，只要求同协议 group-disjoint train holdout 的预测 aggregate relative L2 严格改善，不设最低百分比。界面带、高频带、Marmousi 与运行时间作为诊断指标，不替代主判据。开发期间继续封存 validation/test_id。

## 7. 当前可复现实体

- P1b seed 372 checkpoint：`results/pa_cora_p1b_curriculum_s372_20260902/best.pt`
- checkpoint SHA256：`5d14b93cffc1324406b531b6b2d4a7eea06d9aab57537bb7f4612fea302808d3`
- 边界审计：`results/pa_cora_boundary_projection_s372_20260902.json`
- 边界审计 SHA256：`153781100df18ac216f3c11a9dbb84221408dcbb5600e3e25d2b305ef58dfe47`
- 边界模块：`saved_time_phase_operator_v4/boundary_consistency.py`
- 边界评估器：`scripts/evaluate_pa_cora_boundary_projection.py`

