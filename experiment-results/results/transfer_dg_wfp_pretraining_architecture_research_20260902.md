# Transfer DG-WFP：面向当前声波方程的预训练算法与网络重构方案

日期：2026-09-02

## 结论

下一代预训练模型建议命名为 **Transfer DG-WFP**，全称为：

> Transfer learning with a Discontinuous-Galerkin, Windowed-Fourier Propagator for transient acoustic waves

它不是在当前 v9/P1b 上继续叠加一个弱修正头，而是重构主传播表示：

1. 用直接的有限时间窗 Fourier 系数表示整段 401 帧波场，取消逐帧隐状态递归；
2. 用 WFP 式二维频率局部传播器学习平滑介质中的全局传播；
3. 用频率与系数自适应的 DG 界面散射分支处理反射、透射和多次散射；
4. 保留 MIONet 的多输入思想，但将速度、震源、边界和观测分别编码后再物理融合，不再直接拼接通道；
5. 将 DG 用作端到端传播中的界面消息和匹配监督，不再把局部通量经低阶 skeleton 强行投影成压力修正；
6. 使用全部 2800 条 train 数据并恢复 anomaly 家族，以 HCAIS 做覆盖约束的重要性采样；
7. 推理时不运行平滑 LWC84 背景求解器，目标是保留 A+1 的背景/散射优势，同时消除其部署成本。

该方案是当前证据下最有希望同时提高精度、泛化和推理速度的统一方向。当前阶段只完成研究与设计，没有启动训练。

## 1. 当前问题不是单一超参数问题

### 1.1 当前主干的实测瓶颈

当前选择的 v9 spectral seed 372 为：

- 8 帧真实初始状态；
- 7 个静态条件通道；
- width 64、depth 4、modes 24；
- x/z 两个一维 Fourier 算子相加；
- 约 48.1 万参数；
- 以 warp_r1 全时序为锚点，再做递归残差修正。

train-only group-disjoint holdout 上，v9 onset 窗 aggregate relative L2 为 0.11421，但后续两个窗口升至 0.41849 和 0.48890。对应高频带误差为 0.37135、0.70727 和 0.82422。onset 的 interface-top10 误差为 0.13415，Marmousi interface 误差为 0.20125，而 non-interface 仅约 0.04711。

这表明主要问题是高频、界面散射和跨窗口传播结构，不是顶部边界单点约束。硬自由表面投影虽将 `p(z=0)` 精确置零，但三窗口总误差只改善约 0.00977%。

### 1.2 24 个轴向模态存在带宽缺口，但单独堆模态不是根治方案

数据源频率为 8--30 Hz，保存网格间距为 10 m，实际最低速度约 1425 m/s。最高频率处最短波长约为：

`lambda_min = 1425 / 30 = 47.5 m`。

在 2 km 域内对应约 42 个空间周期，保存网格约 4.75 points per wavelength。当前 `modes=24` 的轴向低通主干不能直接覆盖这部分波数。增大 loss 权重不能恢复网络结构已经截掉的频率。

但已有容量实验也给出了必要的反面约束：uniform/layered 超过 20 模态的真值空间能量低于约 0.5%，Marmousi 超过 32 模态的能量约 0.7%；旧解码器把 modes 从 32 提至 101，只使 all-saved 误差从 0.13381 降到 0.13292。说明“全模态覆盖”应当保留，但不能再用 dense 加宽旧解码器的方式实现。新模型需要同时修复时间相位、二维波数耦合和上游传播表示，额外模态主要负责最后的尖锐波前与界面细节。

### 1.3 当前递归状态不是严格的物理状态

声波方程本身对 `(p, p_t)` 是二阶 Markov 系统，但当前三侧开放边界由物理域外的 CFS-CPML 实现。若把外部 CPML 消元，物理域边界响应带有时间记忆。因此：

- 只用压力帧或一个无物理约束的 hidden state，需要网络同时猜传播状态和边界记忆；
- 长窗口中这种状态误差会累积；
- 8 个历史帧可以缓解，却不能保证得到正确的边界状态。

直接预测整段时间基系数可绕开这个问题：模型学习完整有限时窗的输入到输出算子，不需要在推理时显式恢复 CPML 隐状态。

### 1.4 旧 B2-H 不能作为否定物理结构的证据

旧 B2-H 的固定核只执行顶部自由表面，没有推进 `psi_x, psi_z, phi_x, phi_z` 四个 CPML 记忆变量。它的严格长时 rollout aggregate 约为 0.871，因此被判定失败。

但完整的 201×201 LWC84 + 三侧 CFS-CPML 粗求解器，在 480 条 validation 记录上的均值为 0.04295，family 均值为 uniform 0.01731、layered 0.03270、Marmousi 0.07473。两者相差一个数量级。因此旧 B2-H 的负结果同时混入了边界模型缺失，不能推出“结构保持传播器不可用”。

不过该粗求解器约需 2.8 s/记录，并且最大误差达 0.238；将它直接作为最终部署主干仍不能同时满足速度和 worst-case 精度要求。

### 1.5 当前 DG 路线失败在耦合方式，而不是 DG 原理

已有结果为：

- latent flux 分支相对 control 的均值改善仅 `1.78e-7`，属于数值噪声量级；
- local DtN 的四种子平均 flux relative error 为 0.5823，Marmousi 为 0.8462；
- HCAIS 将均值从 0.58232 改善到 0.58071，证明采样机制有效，但没有解决局部算子容量；
- 56-DOF 固定双线性 skeleton oracle 即使用真实通量，也使 aggregate 从 0.2800 恶化到 1.4881，72/72 条记录全部变差。

所以应淘汰“预测 flux，再用固定低阶 pressure lift 重建全局场”的路径。DG 应进入网络内部，作为界面散射更新和端到端监督。

### 1.6 发现了一个需要先修正的 CPML 域定义问题

数据生成器明确设置 `cpml_outside_physical_domain=true`：保存的 201×201 网格是物理域，CPML 位于其左、右、底三侧之外，输出前已切回物理域并做 2 倍限制。

当前 `dg_interface.py`、local DtN 数据和 skeleton oracle 却使用 `cpml_margin=20`，将保存域左、右、底各 20 个节点当成 CPML 并屏蔽。这会：

- 错误删除物理域近边界的介质界面；
- 使 local dataset 只保留 72 个粗单元，而不是完整物理域的 100 个单元；
- 削弱自由表面附近及侧边界附近的散射监督；
- 使已有 DG 负结果只适用于被错误裁剪的 interior 子域。

后续新实现必须把 `cpml_margin` 从物理网格 face mask 中移除。外部 CPML 只通过边界响应、外延网格或教师标签处理。

## 2. 文献带来的可用机制

### 2.1 Windowed Fourier Propagator

[Windowed Fourier Propagator, 2026](https://arxiv.org/abs/2603.14289) 的关键机制是：波在平滑非均匀介质中的能量主要向邻近空间频率传播，因此无需用 dense 频率耦合；它用局部频率窗口预测传播字典，并显式保持对初始波幅的线性叠加。

这与本项目的缺口高度匹配：

- 当前 axis-factorized FNO 每个方向独立处理，缺少二维邻域波数耦合；
- A+1 已证明平滑速度背景能保留主要传播结构，但真实背景 solve 太慢；
- WFP 的训练分布主要是平滑介质，正好可用于学习 A+1 的 `c_sigma -> P_bg`，而不是直接承担 Marmousi 强间断。

WFP 论文也显示强间断增大时误差会明显上升。因此本方案只让 WFP 负责平滑背景，间断部分交给 DG 分支。

### 2.2 高频偏置与多阶段残差训练

[Frequency Bias and OOD Generalization, 2026](https://arxiv.org/abs/2605.12997) 报告 FNO 对未见高频的退化明显大于 DeepONet，支持保留 MIONet/branch-trunk 条件化并加入显式频率坐标，而不是仅依赖固定低模态 FNO。

[Multi-stage FNO for seismic wavefields, 2025](https://arxiv.org/abs/2503.02023) 用第二阶段 FNO 学第一阶段残差，重点降低高频误差。本项目应采用低中频背景阶段和高频/界面残差阶段分开训练，而不是一次联合优化所有目标。

### 2.3 背景/散射分解

[PICNO, 2025](https://arxiv.org/abs/2507.16431) 将均匀背景波场和速度作为输入，预测散射波场，并报告物理约束对高频和跨介质泛化有帮助。论文也指出 PDE loss 会导致训练波动，因此本方案不让物理项主导，而采用渐进、梯度受限的辅助监督。

本项目自身的 A+1 证据更直接：sigma=2 平滑背景 + 学习散射残差，在早期三记录 held-out 上达到 aggregate 约 0.053，layered 0.082、Marmousi 0.045；但部署时背景求解约 21 s，使总成本高于传统细网格求解。因此新的关键不是再次证明背景分解，而是把昂贵 `P_bg` 求解替换为准确的学习传播器。

### 2.4 DG/FEM 神经算子

- [DG-FEONet, 2026](https://arxiv.org/abs/2601.03668) 用逐单元系数和 SIPG 弱形式处理不连续系数，支持将 volume 与 face loss 分开。
- [NN-MsHDG, 2026](https://arxiv.org/abs/2608.25850) 学习局部 trace-to-flux 算子并做 HDG skeleton 组装，但明确指出高对比下固定低阶 trace space 会产生大误差，局部误差还会被组装放大。
- [Convex Neural Energy Elements, 2026](https://arxiv.org/abs/2608.02036) 给出更强警告：约 1% 的局部场误差也可能经不稳定组装产生 247% 全局误差。结构稳定性必须由参数化保证，不能只看 local regression error。
- [NOEM, 2025](https://arxiv.org/abs/2506.18427) 支持将不同区域作为可复用 neural elements，再通过变分结构耦合。

这些结果与本项目真实通量 oracle 的失败一致：固定低阶 skeleton 和事后组装不适合当前高频、高对比声波。可迁移的是局部 trace、稀疏 face 邻接和稳定散射结构，而不是直接复制椭圆问题的静态 DtN 矩阵。

## 3. 新方法的数学表示

当前教师近似求解：

`p_tt = c(x,z)^2 * Laplacian(p) + q_s(t) * delta(x-x_s,z-z_s)`，

顶部满足 `p=0`，左、右、底通过物理域外的 CFS-CPML 实现开放边界。

先对速度做确定性分解：

`c_bg = Smooth_sigma(c)`，

`j = log(c) - log(c_bg)`。

其中 `c_bg` 保留大尺度传播和主要反射几何，`j` 表示细尺度与间断结构。模型直接预测有限时间窗的复 Fourier 系数：

`P_hat[k] = P_bg_theta[k] + P_DG_theta[k] + P_hf_theta[k]`，

最后：

`p_hat(t) = HardFreeSurface(IRFFT_K(P_hat[k]))`。

这里的 `k` 是 1 s 有限时间窗的 Fourier 基索引。它只是精确的时间表示，不假设每个系数场满足时谐 Helmholtz 方程。项目已有实验已证明，有限时窗、瞬态 Ricker 源和 CPML 会使单个 rFFT bin 不满足标准 Helmholtz 残差；因此不能重新加入已被证伪的 per-bin Helmholtz PDE loss 或固定 Helmholtz resolvent。

保留最低 64 个时间频率 bin 的本地 oracle 已能将完整 401 帧重建到低误差。与 401 步递归相比，64 个复系数场还能避免长时 rollout 漂移。

## 4. Transfer DG-WFP 网络结构

### 4.1 多输入编码器：保留并升级 MIONet 思想

使用四个独立编码器：

1. `E_medium(c_bg)`：多尺度二维介质编码；
2. `E_interface(j, grad(log c), normals)`：界面图和局部高频编码；
3. `E_source(x_s,z_s,f0,t0)`：震源位置、Ricker 频谱及空间相位编码；
4. `E_boundary(d_top,d_side,d_bottom, boundary_type)`：自由表面和开放边界几何编码。

不再把 20 个物理特征简单 concatenate 后交给一个 1×1 projection。建议使用 MIONet 式乘性/双线性融合：

`h_k = Film(E_medium, omega_k) * E_source(k) + Cross(E_medium, E_source, omega_k)`。

其中震源的以下部分是固定解析量，不交给网络学习：

- Ricker 的复时间频谱；
- 源位置对应的空间 Fourier phase；
- source map 的离散低通/限制响应；
- `f0`、`t0` 和 `omega_k` 的无量纲组合。

网络只学习介质如何传播和散射这些已知源分量。

### 4.2 Background-WFP 全局传播器

Background-WFP 只接收 `c_bg`，用三层频率分辨率处理全局传播：

- low band：0--16 周/域；
- middle band：16--40 周/域；
- high band：40--Nyquist，覆盖保存网格完整频谱；
- 每个 band 使用完整二维 `rfft2` 波数坐标，不再分开处理 x/z；
- 每个输入波数只与半径 `r_k` 内的输出波数交互，形成 block-banded/windowed kernel；201×201 物理域覆盖到 mode 100，221×241 外部扩展域的 `rfft2` 网格覆盖到 z 方向 signed Nyquist 110、x 方向 nonnegative Nyquist 120，但不构造 dense 全连接频率矩阵；
- `r_k` 由介质谱宽、传播时间和局部 `f0/c` 条件化；
- 通道混合使用低秩因子，避免 dense `width^2 * modes^2` 参数爆炸；
- 主分支使用正弦/复指数特征或带 Gaussian envelope 的振荡激活，普通 GELU 只用于门控与融合。

建议起始配置：width 96、6 个 residual blocks、二维频谱支持到扩展网格 Nyquist、窗口半径 3/5/7、三尺度下采样 1/2/4。训练采用 48→80→full-Nyquist 的 progressive mode opening；低能模态仍保留探索配额。输出按频率组分头，并由共享的 `omega` 条件化核生成，避免旧 rank-8 全频共享头再次形成秩瓶颈。

时间方向也不设置不可恢复的硬截断：默认重点学习最低 64--96 个 rFFT bin，因为本地 oracle 表明 64 bin 已覆盖主体能量；完整 201 个时间 rFFT bin 作为支持域，通过 source-spectrum/target-energy 门控和 HCAIS 稀疏训练。这样“可表示范围”覆盖完整保存数据，而主要算力仍集中在有能量、有误差的频带。

该分支的主要监督目标是离线 sigma=2 平滑速度教师波场的前 64--96 个复时间系数 `P_bg`，同时用稀疏配额覆盖完整 201-bin 支持域。推理时不运行平滑 LWC84 solve。

### 4.3 Frequency-adaptive DG scattering branch

DG 分支只处理 `j` 中的速度跳变和局部细尺度。首先从保存物理域的所有内部 faces 建图，不设置错误的 20-node CPML margin。

每个 face 的输入包含：

- 两侧 `c_minus, c_plus`；
- `R=(c_plus-c_minus)/(c_plus+c_minus)` 和透射代理；
- 界面法向、曲率和局部入射方向；
- 时间频率 `omega_k`、切向波数和两侧 points-per-wavelength；
- Background-WFP 给出的两侧复压力/相位梯度；
- source 到 face 的相对位置与走时特征。

trace basis 不使用固定 DCT-8 或 P1。按局部频率构造：

- 低频 polynomial/Legendre modes；
- 高频 plane-wave/Trefftz modes `exp(i*k_t*s)`；
- 速度跳变处的 reflected/transmitted paired modes；
- basis 数由局部 ppw 和界面复杂度决定。

face 更新采用“解析散射锚点 + 有界学习修正”：

1. 常密度下的解析反射/透射矩阵给出初始散射；
2. 网络只预测角度、曲率和离散化带来的低秩复修正；
3. 修正矩阵通过 Cayley 或 contractive factorization 参数化，保证内部 face 不凭空放大能量；
4. 相邻单元接收成对的守恒消息；
5. 使用 2--4 轮 scattering-order message passing 表达多次反射，而不是沿 401 个时间点递归。

DG 分支直接输出 `P_DG` 并参与全场 loss。它不再经过独立的最小二乘 skeleton pressure lift，因此局部误差不会脱离全场目标后被放大。

### 4.4 高频残差分支

在 Background-WFP + DG 后增加一个较小的 high-frequency residual head：

- 输入为当前复系数、Haar/多小波高通介质特征和 DG face scatter；
- 只预测 32--Nyquist 的残差，并保留完整频谱支持；
- 使用局部 3×3/5×5 卷积和小波上采样，不做低模态 Fourier 截断；
- 采用第二阶段独立训练，再联合微调。

该分支对应当前 high-band 0.37--0.82 的主要误差，并落实多阶段 FNO 的机制。

### 4.5 物理域外 CFS-CPML 条件

新模型不再把 CPML 当作物理域内的 mask，而是显式恢复教师的外部扩展域：

- 物理细网格为 401×401，外部 CPML 左/右各 40、底部 40，完整细网格为 441×481；
- 对应 10 m 保存尺度，物理域为 201×201，外部 CPML 左/右各 20、底部 20，完整扩展域为 221×241；
- 顶部没有 CPML，始终硬执行 pressure-release `p(z=0)=0`；
- 外部最左、最右和最底边界与教师一致置零，波在到达该边界前由 CPML 衰减；
- CPML 中的速度由物理域边缘值向外复制，与教师 `_extend_velocity` 一致。

为扩展域构造并输入教师完全一致的固定 profile：

- `sigma_x, sigma_z`；
- `kappa_x, kappa_z` 及其倒数；
- `alpha_x, alpha_z`；
- ADE 离散系数 `a_x, a_z, b_x, b_z`；
- `active_x, active_z`、physical-domain mask 和三种边界类型 mask。

参数保持当前数据协议：物理厚度 200 m、`target_reflection=1e-8`、polynomial order 3、`kappa_max=3`、`alpha_max=pi*8 rad/s`，内部时间步 0.125 ms，每个 2.5 ms 保存间隔 20 个子步。`sigma_max` 仍由每个实例的参考速度、厚度和目标反射系数确定。

CPML-ADE 的四个记忆变量为 `psi_x, psi_z, phi_x, phi_z`。其固定更新结构保持为：

`psi_d(n+1) = b_d*psi_d(n) + a_d*D_d p(n+1)`，

`Dtilde_d p = inv_kappa_d*D_d p + psi_d`，

`phi_d(n+1) = b_d*phi_d(n) + a_d*D_d(Dtilde_d p)`，

`L_cpml,d p = inv_kappa_d*D_d(Dtilde_d p) + phi_d`。

推荐采用“扩展域直接预测 + 固定 ADE 监督”的实现：

1. Background-WFP 在 221×241 扩展域产生压力的时间 Fourier 系数；
2. 一个低宽度 CPML state head 只在 active strips 预测四个 memory trajectories，不在完整物理域浪费通道；
3. 离线重新生成扩展域 pressure 与四个 ADE memory 的教师缓存；
4. 训练时同时约束 pressure、memory update、CPML acceleration 和 physical/CPML 入口的 pressure/normal-flux 连续性；
5. 推理输出最终只裁取 `z=0:201, x=20:221` 的 201×201 物理域。

由于主模型直接预测整个有限时间窗，CPML memory 不需要成为 401 步全场递归状态。四个 ADE 状态只用于外层条件和一致性监督，从而保留边界物理而不重新引入全域时间步进成本。

边界损失建议为：

`L_boundary = L_ADE_state + L_CPML_accel + L_entry_flux + L_outer_zero + L_reflection`。

- `L_ADE_state` 检查四个 memory 的因果更新；
- `L_CPML_accel` 检查 CPML 拉伸导数产生的加速度；
- `L_entry_flux` 保证物理域与外部层入口的压力和法向通量连续；
- `L_outer_zero` 对扩展域最外边界硬置零；
- `L_reflection` 比较向内返回能量与教师，防止网络用非物理反射拟合内部场。

如果完整扩展域 WFP 的推理成本偏高，第二阶段再蒸馏为边界专用的被动状态空间算子；第一版先使用真实 221×241 外域，避免在结构验证前再次近似掉 CPML。

## 5. 训练目标

总目标建议为：

`L = L_field + lambda_spec*L_spec + lambda_bg*L_bg + lambda_DG*L_DG + lambda_grad*L_grad + lambda_energy*L_energy + lambda_boundary*L_boundary`。

### 5.1 主目标

- `L_field`：按记录计算完整时空 relative L2，近零时段使用已有 onset-energy floor；
- `L_spec`：对 64 个复时间系数及 low/mid/high 空间频带分别归一化；
- 最终 checkpoint 仍按完整、固定、group-disjoint 集合的全场 relative L2 选择。

### 5.2 分解监督

- `L_bg`：Background-WFP 对 sigma=2 平滑教师的复系数回归；
- 同时监督 `P_true-P_bg` 与总场，避免残差很小时网络输出噪声；
- uniform 家族的真实散射近零，使用绝对能量与 relative loss 混合，避免近零分母产生假性巨大误差。

### 5.3 外部 CPML 监督

- 在 221×241 扩展域上监督 pressure，在 active strips 上监督四个 ADE memory；
- 使用教师内部 0.125 ms 状态或其精确 20-substep 聚合标签，不用物理域内的伪 CPML margin；
- 物理域 full-field loss 与 CPML loss 分开归一化，防止外层低能量区域被主场能量淹没；
- CPML loss 只约束开放边界，不参与内部 material-interface 的 HCAIS 配额。

### 5.4 匹配 DG 监督

- 在 teacher pressure 上用与数据生成一致的空间 stencil 构造 face normal derivative；
- 同时监督 pressure trace、normal flux、reflection/transmission action 和最终 pressure correction；
- face loss 只做辅助，不单独决定 checkpoint；
- 使用 HCAIS 在 family、frequency、face type、interface strength 和 trace leverage 间平衡采样；
- 对非均匀抽样保留 inverse-probability correction。

### 5.5 物理约束的使用边界

不使用：

- 有限时间窗 rFFT bin 的 Helmholtz 强残差；
- 与教师不匹配的 Q1 saved-grid 强制残差；
- 未表示外部 CPML 状态的裸 LWC84 rollout loss；
- 只降低通量残差、不检查全场误差的目标。

可以使用：

- 与教师完全匹配的离线 face flux 标签；
- 顶部硬 Dirichlet；
- DG 散射矩阵的守恒/被动性约束；
- 从细网格教师直接保存的 macro-step 增量或边界 traction 标签；
- 仅作为弱辅助的时域离散一致性。

物理 loss 在前 20% 训练中从 0 线性升高，并限制其梯度范数不超过 `L_field` 梯度的 20--25%，防止复现 physics-cond/PICNO 类优化不稳定。

## 6. 数据与采样重构

### 6.1 立即可用的数据

当前数据有 2800 条 train、600 validation、600 test_id 和 3 条 canonical OOD，包含 uniform、layered、anomaly、Marmousi。当前 v9 只使用 240 条拟合记录，且主要评估三族，远未利用现有离线数据容量。

第一轮正式预训练应：

- 使用全部 2800 train；
- 恢复 anomaly 家族；
- 以 group_id 隔离训练与选择；
- validation 用于模型开发与早停；
- 若 validation/test_id 都参与设计，必须另生成新的最终 blind test，不能再把旧 test_id 称作独立测试。

### 6.2 需要新增的离线标签

预训练代价不设上限，因此建议为 train 记录生成：

- sigma=2 平滑速度 `c_bg`；
- 对应的 `P_bg` 64-bin 复系数；
- `P_true-P_bg` 复散射系数；
- 所有内部 face 的 teacher pressure/normal-flux trace；
- 界面法向、曲率、局部 ppw、入射角代理；
- 221×241 外部扩展域的 pressure、CPML profiles 和 active masks；
- active strips 上的 `psi_x, psi_z, phi_x, phi_z` 保存时刻状态；
- 物理/CPML 入口的 pressure/normal derivative 标签；
- 必要时保存少量 fine-grid macro-step 标签，而不是保存全部 401×401 子步场。

### 6.3 新增介质覆盖

现有数据应补充可控的界面 curriculum：

- 倾斜层、弯曲层、断层、薄层和楔形体；
- 多尺度 anomaly、尖角与高曲率界面；
- 速度对比、界面密度、源深和入射角联合分层；
- f0 与最小 ppw 联合覆盖；
- 自由表面附近及物理侧边界附近的反射/散射样本。

HCAIS 使用 40% coverage、40% difficulty、20% leverage 的防御混合。已有 local DtN 配对结果虽只改善 0.278%，但三个 family 均值同时改善且 ESS/overhead 通过，足以保留这一采样器。

## 7. 四卡训练方案

### 7.1 分阶段训练

1. **Stage A：Background-WFP distillation**
   - 只训练 `c_bg + source -> P_bg`；
   - 物理域空间频谱支持到 mode 100，扩展域支持到 z/x Nyquist 110/120，时间频谱支持完整 0--200 rFFT bins；
   - 先训练主体 64 个时间 bin 和 48 个空间模态，再按 48→80→full-Nyquist progressive opening；
   - 高频低能区使用稀疏频率窗口和共享参数，不使用 dense 全模态矩阵。

2. **Stage B：DG scattering pretraining**
   - 冻结 Background-WFP；
   - 训练 adaptive trace basis、稳定 face scattering 和 2--4 轮消息传递；
   - 同时优化 scatter field、trace flux 和 full-field loss。

3. **Stage C：high-frequency residual**
   - 冻结低频主干，训练小波/局部卷积高频头；
   - 重点采样 high f0、low ppw、strong interface。

4. **Stage D：joint fine-tuning**
   - 全网络低学习率联合训练；
   - 主损失始终是完整全场误差；
   - 物理辅助项按梯度比例受限。

### 7.2 资源配置

- 四张 RTX 4090 D 全部用于 DDP；
- 每卡 micro-batch 1--2 records，时间频率按 8 或 16 bins 分块；
- gradient accumulation 形成有效 batch 16--32 records；
- 网络卷积用 BF16，FFT 和 complex reduction 保持 FP32；
- activation checkpointing 只用于 6 个 WFP blocks；
- 每 GPU 4--8 个 shard-local workers，禁止四个 rank 随机读取同一个大 VDS 热点；
- 保存 `best.pt` 和 `latest.pt`，中间 epoch checkpoint 原子覆盖，继续遵守只保留最新检查点策略；
- 推理一次编码介质；默认激活由 Ricker 频谱与介质风险确定的时间/空间模态，必要时可开启完整模态；最后一次 irFFT，不执行数值背景 solve。

## 8. 最小可归因验证顺序

### E0：协议和实现审计

- 移除保存域上的错误 CPML margin；
- 确认 201×201 全部是物理域；
- 建立 221×241 保存尺度外部 CPML 扩展域，核对 441×481 教师细网格；
- 确认 top hard boundary、四个 ADE memory 和外部 CPML teacher contract；
- 用教师轨迹验证 20-substep ADE 聚合、入口通量和裁剪结果；
- 建立 all-2800 train census 与 anomaly 覆盖。

### E1：只验证 Background-WFP

同一组 sigma=2 `P_bg` 教师，对比：

- 旧 U-Net/rank-8 背景预测；
- full-2D FNO；
- Transfer WFP。

只要求同协议 holdout 全场/背景误差严格改善，不设置任意百分比门槛。必须同时报告 64-bin、high-band、boundary 和 runtime。

### E2：加入解析 DG scattering

- 只用解析 reflection/transmission anchor，不启用 learned correction；
- 检查 interface 与 Marmousi 是否改善；
- 验证能量不增和 equal/opposite face scatter。

### E3：加入 learned adaptive DG correction

- plane-wave trace 与固定 DCT/P1 做配对；
- HCAIS 与 uniform sampler 做配对；
- 端到端 full-field error 是唯一晋升指标，local flux 只作诊断。

### E4：加入 high-frequency residual 并联合训练

- 对比 24、48、80、full-Nyquist spatial modes，并区分“旧 dense 扩模态”与“WFP 稀疏窗口全支持”；
- 对比单阶段和多阶段高频残差训练；
- 最终检查 aggregate、family、time、frequency、interface、boundary、nonworse 和 runtime。

在 E1 没有证明 WFP 能替代昂贵 `P_bg` 前，不启动完整 Stage B--D 大训练。这是当前最关键、最便宜的结构性 go/no-go。

## 9. 与在线自监督微调的接口

预训练完成后，在线阶段只开放：

- DG scattering 的低秩修正；
- 每个频带的相位/幅值 LoRA；
- source calibration；
- 可选的观测融合 adapter。

冻结 Background-WFP 主干和硬边界。在线目标使用可获得观测、匹配的时域弱物理约束、DG face consistency 和 passivity。HCAIS 选择信息量高的时间、频率和界面；不依赖未来真值。这样“Transfer”对应离线全局传播知识向具体实例的低维迁移，而不是在线重训整个求解器。

## 10. 最终判断

最值得投入的根本优化不是继续扩大当前递归 FNO，而是：

`A+1 背景/散射证据 + WFP 快速平滑传播 + adaptive DG 界面散射 + MIONet 多输入编码 + 全支持稀疏时间合成`。

该组合每一部分都对应一个已测瓶颈：

- WFP 替代 A+1 的昂贵背景 solve；
- 以 64--96 个主时间 bin 为重点、完整 201-bin 为支持域的直接合成消除递归状态与长时漂移；
- full-2D frequency windows 提供到 Nyquist 的完整模态支持，并补足 modes=24 和轴分解缺口；
- adaptive DG scattering 处理 WFP 在强间断介质中的弱点；
- 稳定 face 参数化避免 local error 经 skeleton 放大；
- MIONet 分离速度/震源，避免 naive physics-channel concatenation；
- all-2800 + anomaly + HCAIS 改善数据覆盖与采样效率。

下一步应先实现 E0 和 E1，不应直接重跑 v9，也不应进入旧 local DtN 的第三步。
