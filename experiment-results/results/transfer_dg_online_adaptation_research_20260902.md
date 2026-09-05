# Transfer DG：离线神经算子到在线间断 Galerkin 自监督微调

## 方法命名与定位

新方法正式命名为 **Transfer DG**。

其核心不是在线运行一次完整 DG 正演，而是把离线预训练神经算子的解迁移到当前速度模型，再用间断 Galerkin 的单元残差和界面数值通量，对一个低维、局部化的修正空间做快速自监督投影：

`offline pretrained operator -> instance prediction -> DG residual projection -> corrected wavefield`。

离线阶段可以昂贵；在线阶段必须只读取速度、震源、时间网格、允许的 onset 观测和父模型预测，不能读取未来真值。

## 1. 为什么应该从现有 Q1-FE 转向 DG

项目已有连续 Q1-FE weak adapter，但平衡训练面板已经拒绝它：

| 方法 | 12 条记录 aggregate | 相对父模型变化 | nonworse |
|---|---:|---:|---:|
| parent | 0.13530719 | - | 12/12 |
| observed-only | 0.13792092 | -1.932% | 3/12 |
| Q1-FE weak | 0.13776354 | -1.815% | 4/12 |

Q1-FE 项比 observed-only 略好，但仍使总体预测变差。当前实现存在五个与界面散射直接相关的限制：

1. 连续 Q1 空间将速度突变单元平均化，没有显式左右迹与界面数值通量。
2. 只对组装后的节点残差优化，不能区分单元体误差和界面反射/透射误差。
3. 修正变量只是 64 个全局 decoder channel scale，无法在特定界面附近局部修正。
4. 多条记录的 `source_off_frame` 接近 61，可用弱残差时间点过少。
5. 当前只有压力场，无法直接构造能量稳定的一阶声学 Riemann 通量。

因此，Transfer DG 不能只是把 Q1-FE 损失改名，而要更换离散空间、状态变量和在线修正参数化。

## 2. 文献调研结论

- [DG-FEONet](https://arxiv.org/abs/2601.03668) 用 SIPG 弱形式残差训练逐单元解系数，目标正是不连续系数和非光滑解。它直接支持“神经算子输出 + DG 单元/界面残差”的组合，但该工作是数据自由的全模型训练，并非实例级快速迁移。
- [LRNN-DG for diffusive-viscous wave equation](https://arxiv.org/abs/2305.16060) 使用局部神经网络和 space-time DG 连接子域，说明 DG 数值通量可把多个局部神经表示拼成时间相关波场。
- [Energy-stable DGSEM-PML for acoustic waves](https://arxiv.org/abs/1802.06388) 为异质声学介质构造物理上风通量，并指出 PML 稳定性要求把界面与边界过程一致地扩展到 PML 辅助方程。Transfer DG 第一版应排除 CPML 内区；只有模型同时输出 PML 辅助变量后才能把 CPML 纳入 DG 残差。
- [hp-VPINNs](https://arxiv.org/abs/2003.05385) 使用局部多项式测试函数和域分解，表明分部积分可以降低正则性要求，并把优化集中到局部困难区域。
- [PINO](https://arxiv.org/abs/2111.03794) 将数据与更高分辨率 PDE 约束结合，支持离线监督预训练、在线物理约束的双阶段结构。
- [One-shot transfer PINN](https://arxiv.org/abs/2110.11286) 和[离线/在线 transfer PINN](https://arxiv.org/abs/2205.07731) 支持冻结大部分预训练表示、只迁移小部分参数以加速新实例求解。
- [Hybrid FE/neural solver](https://arxiv.org/abs/2307.00947) 采用粗网格解加局部神经细尺度修正，支持将 Transfer DG 的可训练自由度限制在高残差界面 patch。

检索中尚未发现与以下完整组合相同的方法：离线声学神经算子、实例级低维迁移、DG 界面通量、自由表面/CPML 协议、稀疏 onset 观测和候选先封存后评估。该组合可以形成我们的主要方法贡献，但正式论文仍需做更系统的专利与引文追踪，不能仅凭本轮检索宣称绝对首创。

## 3. Transfer DG 的数学结构

### 3.1 一阶声学状态

离线模型由只输出压力扩展为辅助多头：

`y = (p, u_x, u_z)`，后续 CPML 版本再加入记忆变量 `psi_x, psi_z`。

在非 CPML 区域使用常密度一阶声学系统：

`(1/K) * partial_t p + div(u) = q_p`

`rho * partial_t u + grad(p) = q_u`

其中 `K = rho*c^2`。在速度间断面上，DG 不强制网络全局光滑，而是允许每个单元具有独立左右迹，并用阻抗加权声学 Riemann 通量连接 `p` 与法向速度 `u_n`。这正是反射和透射发生的位置。

### 3.2 在线低维修正

离线阶段学习一组 correction basis `B_j(x,t;c,s)`，按低/中/高频、界面方向和局部 patch 分组。新实例的父预测记为 `y_parent`，在线候选为：

`y_alpha = P_boundary(y_parent + sum_j alpha_j B_j)`。

只优化系数 `alpha`，主干、频谱层和 basis 全部冻结。目标为：

`min_alpha ||R_volume(y_alpha)||_W^2 + lambda_flux ||R_flux(y_alpha)||_F^2 + lambda_obs ||H y_alpha-y_obs||^2 + lambda_ridge ||alpha||^2`。

当 basis 对 `alpha` 线性且采用一阶线性声学 DG 时，该目标是带正则的二次问题，可以通过小型 Cholesky/QR 一次求解，而不是在线反向传播整个网络。这有利于满足 10x 端到端速度要求。

### 3.3 单元和界面项

- **Volume residual**：每个 space-time 单元内积分一阶声学守恒方程。
- **Interior flux**：在速度间断面使用阻抗加权上风通量；分别报告压力迹和法向速度迹的缺陷。
- **Free surface**：顶面使用压力释放边界 `p*=0`，并保留现有硬投影。
- **CPML**：v1 排除左、右、底 20 格 CPML；v2 只有在预测 CPML 辅助状态并实现能量稳定通量后才进入损失。
- **Source region**：源激励期间使用已知 Ricker source 进入体残差，不能像旧 Q1-FE 一样简单把几乎整个时间窗排除。

## 4. 自适应局部化与高频

在线活动单元只能由部署合法信息选择：

- `|grad log(c)|` 和速度反射系数代理确定介质界面；
- 父预测的 DG volume/flux defect 确定物理高缺陷区；
- f0 与 points-per-wavelength 确定需要激活的高频 basis；
- onset 观测邻域确保修正与已观测波场一致。

活动集合在读取未来真值前冻结。每个界面 patch 只保留少量多项式/小波 correction basis，从而同时获得局部表达和小规模在线线性系统。

## 5. 与离线预训练的关系

Transfer DG 的完整方法分为两部分：

1. **Transfer DG-Pretrain**：当前正在进行的 A1 边界传播，随后增加多尺度高频和界面散射支路；离线训练 pressure/velocity 辅助头和 DG correction basis。
2. **Transfer DG-Adapt**：新实例上冻结预训练主干，使用 onset 观测与 DG 自监督残差求解低维系数。

这仍属于迁移学习：离线模型提供跨介质共享表示，DG 在线阶段只把共享表示投影到当前实例的离散物理解空间。

## 6. 实现与验证顺序

### TDG-0：离散算子契约

- 常数场 volume residual 为零；
- 解析平面波在均匀介质中收敛；
- 两层介质解析反射/透射通量缺陷接近零；
- 自由表面反相压力满足边界通量；
- DG 离散能量在无源内域非增长；
- CPML 区在 v1 中严格不参与目标。

### TDG-1：训练域配对归因

在同一批 train-only 记录上比较：

- parent；
- onset observed-only；
- 旧 Q1-FE；
- Transfer DG volume-only；
- Transfer DG volume + interface flux。

所有候选必须在读取未来真值前序列化并哈希。主判据按用户要求仅为同协议 aggregate relative L2 严格改善，不设置最低改善百分比；同时报告每个 family、界面带、高频带、nonworse 和完整在线时间。

### TDG-2：group-disjoint train confirmation

冻结 basis 数、单元大小、penalty、ridge、活动单元数和求解步数，在新的 group-disjoint train split 确认。通过后才允许一次完整 validation；test_id 保持封存到最终版本。

## 7. 风险与控制

- DG residual 下降不保证真实误差下降，旧 Q1-FE 面板已经证明这一点；必须用 train-only paired future diagnostic 做归因。
- 如果只用压力近似速度通量，可能产生伪反射；因此正式版本需要辅助速度头，压力-only 版本只能作为拒绝性诊断。
- SIPG penalty 不能在 validation 上调；应根据网格尺寸、局部波速/阻抗和多项式阶数预先计算或在 train-only 校准。
- 完整 DG 正演可能破坏速度目标；Transfer DG 只求解低维 correction 系数，不在线求解全网格 DG 自由度。
- 只有同协议 validation/test_id 精度与端到端 10x 速度同时达标，才能声称最终目标完成。

