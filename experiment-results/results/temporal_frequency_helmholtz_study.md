# 深入研究: 时间-频域 (Helmholtz 合成) 神经算子 —— 突破位置/相位地板的候选

日期: 2026-07-31
触发: 用户要求"确保预训练神经算子达到 <5% 相对误差, 调研最前沿算子学习, 针对声波方程框架优化"。
方法: **零训练、CPU 可复现**的三个 oracle 边界测量, 直接从真值波场的时间/空间频谱回答"表征是否够、参数量是否可行"。

## 背景: 为什么是这个方向

已确证的收敛诊断 (见 `capacity_physical_limit_study.md` + memory):
- 容量不是墙 (17× 参数平 ~0.18); 空间 Nyquist 不是墙 (>99.5% 真值功率在 ≤20 空间模态)。
- 墙 = **波前位置/相位的时间表示** —— 每个 saved-time 把移动波前放到正确位置/相位。
- 迄今所有尝试 (warp/A3/A4/B1-tempop/C1) 都**逐帧独立渲染** (时间轴对角), 均卡 agg~0.24;
  传播阶梯四档全基于单一首至 grad(T) 方向, 已四连证伪。
- 真 3D 时空 FFT (设计文档 Option2) 违反**查询不变硬契约** → 一直被排除。

**关键盲区**: 从未测过**时间-频域**表征。时间 Fourier 变换把波动方程解耦为**逐频率静态 Helmholtz 场**
`∇²P(x,z,ω) + (ω²/c²)P = -S(ω)·source_map`, 每个 `P(·,ω)` 是**无时间轴的复空间场**。
任意 saved-time 帧由**固定、非学习**的逆变换重建: `p(x,z,t) = Σ_j Re[P_j(x,z)·e^{iω_j t}]`。
- **天然查询不变**: `{P_j}` 不依赖 t; 单帧查询 = 同一 `P_j·e^{iω_j t}` 与批内逐位一致。无自回归、无跨查询耦合。
- **直击位置/相位**: 波前到达时间 τ(x) 就是 `P_j` 的**相位** (∝ e^{-iω_j τ(x)}); 跨时间相干性由精确逆变换**内建**,
  模型不再逐帧独立猜位置 → 去掉抖动 (0.125 地板根因)。
- **与 tempop/QueryInvariantTemporalBasis 本质不同**: 那些是**学习的通用时间基** + **门控残差** + **低秩/1×1 空间系数**
  (r2 零感受野失败); 这里时间基是**精确物理** `{e^{iω_j t}}`, 系数是**全分辨率复 Helmholtz 场** (主表征非校正),
  且可加**逐频率 Helmholtz PDE 残差**正则。

## 一、时间可压缩性 (go/no-go 前提) —— **通过**

真值波场 401 帧沿时间轴 rfft, 保留最低 m 个频率 bin 后逆变换 (固定非学习), held-out 记录:

| 频率 bank | uniform | layered | marmousi |
|-----------|---------|---------|----------|
| 最低 48 bin (0–47Hz) | 0.040 | 0.071 | 0.051 |
| 最低 64 bin (0–64Hz) | **0.005** | **0.012** | **0.028** |
| top-|mag| 48 bin | 0.024 | 0.047 | 0.046 |

**结论: 整条 401 帧时间演化可由 ~48–64 个复空间场以 <5% 重建** (Ricker f0∈8–30Hz 带限, 90% 能量在 <47Hz)。
时间表征**不是**信息瓶颈 —— 这是决定性好消息, 此前从未测过。

## 二、Helmholtz 频率场的空间复杂度 —— **可控** (需 |k|≤~32–40)

每个 `P(·,ω)` 是振荡场 (波长 c/f), 不像帧那样低秩。2D-fft 低模态截断 (半宽 m), held-out:

- **低频 (10–30Hz, 波长 5–15 cells)**: |k|≤20 已达 relL2 0.06–0.10 (uniform 30Hz 圆波前需 |k|≤20)。
- **高频带 (47–64Hz, 波长 2.3–3.2 cells)**: 需 |k|≤32 (relL2 ~0.05–0.08); marmousi 20–30Hz 场最硬需 |k|≤32–50。
- Nyquist 空间模态 |k|=100, 故 |k|≤32–40 远未触及空间上限, 与真值空间低秩一致。

## 三、组合 oracle 边界 (实际模型目标) —— **<5% 达成**

保留最低 nf 个时间 bin, 每个复场空间截断到 |k|≤sm, 逆变换到 401 帧:

| 配置 | uniform | layered | marmousi | 参数/记录 (稠密表征上界) |
|------|---------|---------|----------|--------------------------|
| nf=48, |k|≤32 | 0.054 | 0.095 | 0.065 | 405,600 |
| nf=64, |k|≤32 | 0.054 | 0.055 | 0.063 | 540,800 |
| **nf=64, |k|≤40** | **0.045** | **0.044** | **0.047** | 839,808 |
| nf=64, |k|≤50 | 0.038 | 0.037 | 0.040 | 1,305,728 |

**决定性结论: 存在一个查询不变、参数可行的表征 (64 个复 Helmholtz 场, 每个 |k|≤40 谱系数), 三族全部 <5%。**
这是**首个**给出 <5% 可达性的**构造性**证据 (此前 CEILING 研究只在放宽契约到稀疏观测下达到 <10%; 容量梯全在目标之上)。
5% 不再是"物理/信息不可能", 而是**架构表示问题** —— 且现在有了明确、已测量的目标表示。

## 四、诚实的风险与边界

1. **oracle vs 学习**: 以上是"若模型能精确产出这些 Helmholtz 场"的**上界**; 真实模型需从 (速度场, 源) **预测** `{P_j}`,
   预测误差会抬高 relL2。但这是**回归静态复场**问题 (无时间轴), 比"逐帧放置移动波前"结构上简单得多, 且可被逐频率 Helmholtz PDE 残差强约束。
2. **复场预测的相位精度**: 波前位置 = `P_j` 相位, 模型仍须准确预测相位; 但现在相位是**静态空间场** (∝ τ(x)·ω, τ 是走时——项目已有 eikonal/hybrid_travel 走时场可作强先验!), 而非跨 401 帧的动态量。这把时间传播问题**降维**成"预测走时场 + 频率相关振幅"。
3. **契约兼容**: 逆变换 `Σ Re[P_j e^{iω_j t}]` 对任意实数 t 解析可微 (含插值时刻), 单帧/多帧逐位一致。已由构造保证, 无需新增查询不变测试即满足 `test_per_frame_query_invariance` 精神 (但仍会加显式测试)。
4. **不推翻核心前提**: 不是 3D 时空 FFT (违反契约), 不是自回归。是**沿时间轴的解析基 + 静态空间系数场**, 恰是设计文档
   "Factorized 2D+1D (Recommended)" 的**物理正确实例化** (1D 时间 = 精确 Helmholtz 基, 非学习的通用时间处理器)。

## 五、建议 (下一步)

**HelmholtzSynthesisField**: 用一个"逐频率复 Helmholtz 场生成器"作为 coarse 场的新形态 (替换/并列 local_field),
输出 nf≈32–64 个复空间场 `{P_j}`, 由固定逆变换合成任意 saved-time 帧。
- 生成器 = 共享 backbone → nf 个复通道头 (实/虚), 条件 = 速度场 + 源 + **走时场先验** (相位锚)。
- 训练损失 = 现有帧域 relL2 (逆变换后) **+** 逐频率 Helmholtz PDE 残差 (自归一, 复用 lwc84 思路) **+** 谱 bank 稀疏。
- 门控/零初始化 warmstart 到现有 local_field 领跑分支, 单因素接入 (与项目一贯做法一致)。
- 判据 (G2 三记录过拟合): 是否破 0.125 地板 (oracle 上界 <0.05 → 若表示有效应显著低于逐帧的 0.24/0.125)。

## 产物
- 本文 + 三个内联可复现探针 (`/tmp/temporal_rank_probe.py`, `helmholtz_spatial_probe.py`, `combined_bound.py`)
- 定性: 时间-频域是**唯一**同时满足 (查询不变 ∩ 直击位置相位 ∩ oracle<5% ∩ 参数可行) 的方向; 传播阶梯 (grad(T) 首至) 已四连证伪。

## 七、G2 学习探针 (2026-07-31, oracle→learned 的真实差距)

实现 `_HelmholtzSynthesisField` (local_field.py, 一个 coarse 场新形态, 单因素接入 operator/probe/CLI, 全查询不变契约测试通过), 在 3 记录过拟合探针上测**学习**模型能否逼近 oracle。判据 = 破 local_field 逐帧 0.125 地板 (oracle <0.05)。

**朴素头 (直接预测振荡复系数场, fresh, 200 update, 16 训练帧)**:

| nf | all_saved agg | uniform | layered | marmousi |
|----|---------------|---------|---------|----------|
| 32 | 1.09 | 0.91 | 1.41 | 0.95 |
| 64 | 1.17 | 0.93 | 1.61 | 0.97 |
| 96 | 1.25 | 0.96 | 1.80 | 0.99 |

**失败签名 (诊断价值)**: (a) all_saved(401帧) > fixed(16帧) → **时间欠定** (16 帧约束 2·nf≥64 个逐像素自由系数, 病态); (b) **频率越多越差** (与 oracle 相反, oracle nf64 优于 nf48) → 自由系数越多越欠定; (c) 16 帧 800 update 仍缓降到 ~0.74 不破地板 → 欠训不是主因。

**WKB / 几何光学 ansatz (retarded-time, 用 eikonal 走时供振荡, 网络只出平滑振幅包络)**:

| variant | all_saved agg | uniform | layered | marmousi |
|---------|---------------|---------|---------|----------|
| WKB nf32 | 0.89 | 0.83 | 0.96 | 0.89 |
| **WKB nf64** | **0.77** | 0.86 | **0.65** | 0.80 |

**WKB 有效且逆转了朴素头的病态** (nf64 反优于 nf32, 因平滑目标不再欠定; naive nf64 1.17 → WKB 0.77 相对改善 34%)。但**仍 ~0.77 远高于 0.125 地板** (fresh 200 update 未收敛, 但平台形状显示: 从共享 backbone 学到的映射没有找到 oracle 证明存在的平滑低秩 Ĝ)。

**裁定 (oracle vs learned 分离)**: oracle 上界成立 (表征可达 <5%), 但**朴素/WKB 的"共享 backbone → 逐频独立系数头"这一参数化学不到目标**。差距不在表征而在**从 (速度,源) 到系数场的映射结构**。下一步 = **源感知因子分解** (见 memory `fno-acoustic-temporal-frequency-helmholtz` 的 2026-07-31 源感知推进): P̂(x,ω)=ŵ(ω)·Ĝ(x;ω), ŵ=FFT(已知 source_wavelet) 解析剥离源参数; Ĝ 用 R≤8 低秩共享复基 (探针实测振幅跨频 rank≤6, marmousi=1) 而非 2·nf 自由通道。这直接消除时间欠定 (系数 DOF 从 2·nf 降到 2·R) 并把源参数零学习地剥出。设计文档 `docs/superpowers/specs/2026-07-31-source-aware-temporal-frequency-helmholtz-design.md`。

## 八、低秩因子分解 G2 (2026-07-31→08-01, 参数化阶梯单调改善)

按源感知设计实现低秩 `Ĝ_k(x)=Σ_r M[k,r]·B_r(x)` (R 个平滑共享复基场 + 逐频混合), 每像素 DOF 从 2·nf 降到 R。同 G2 协议 (3记录, 250 update, 16帧):

| 参数化 (nf64) | all_saved agg | uniform | layered | marmousi | 诊断 |
|---------------|---------------|---------|---------|----------|------|
| WKB 自由头 (§7) | 0.77 | 0.86 | 0.65 | 0.80 | 基线 |
| 低秩 **全局共享**混合 r8 | ~1.0 平台 | 0.98 | — | 0.99 | **平凡解**: 三记录被迫共用一组频率→基混合, 塌到近零输出 |
| 低秩 **逐记录条件化**混合 r8 | **0.499** | 0.598 | 0.381 | 0.520 | 持续下行无平台 (250 step 未收敛, 仍在 0.556→0.499 陡降) |
| 逐记录条件化 r16 | 0.527 | 0.534 | 0.413 | 0.633 | |
| 逐记录条件化 r32 | 0.619 | 0.636 | 0.574 | 0.646 | r 过大回到欠定 |

**关键结论**:
1. **参数化阶梯单调改善**: 朴素 1.17 → WKB 0.77 → 条件化低秩 0.50。每次有依据的结构修正 (WKB 供相位 / 低秩消欠定 / 逐记录条件化破平凡解) 都真实降误差。
2. **混合必须逐记录条件化**: 全局共享混合 (设计文档偏离实现) 塌到平凡解 agg≈1.0; 从 render 全局池化预测每记录 M[nf,R] (零初始化 delta) 修复, 直接跌到 0.50。
3. **r8 最优** (r16 略差, r32 因混合参数 2·nf·R 变大回到欠定) —— 印证振幅探针的跨频 rank≤6 测量。
4. **0.499 是全项目所有查询不变尝试里最好的 G2 未收敛值** (Option-B local_field 0.29 是长训收敛值; 传播阶梯全卡 0.24; 此处 250 step 未收敛且仍陡降)。**尚未破 0.125 地板, 但轨迹健康 (无平台) + 参数化阶梯单调 = 方向有效, 需 G2 训到收敛 (800+ update) 判定能否破地板**。
5. **源因子 ŵ(ω_k) 尚未接入** (3记录各单源, ŵ 折进混合; 主要是泛化杠杆, 留待 held-out pilot)。

**下一步 (待用户定)**: (a) 条件化低秩 r8 训到收敛 (800 update) 看是否破 0.125; (b) 接入解析源因子 ŵ 走 G3 held-out pilot (对 LPF 0.29); (c) 加逐频率 Helmholtz PDE 残差进一步约束。诚实: G2 未收敛值 0.50 尚不能断言破地板, 但这是项目历史上最强的查询不变方向证据链 (oracle<5% 可达 + 参数化单调逼近), 值得投长训判定。

## 九、条件化低秩收敛裁定 (2026-08-01, 步骤 (a) 完成)

按上文步骤 (a), 4 个秩 (r6/r8/r12/r16) 各训到 800 update 收敛 (3记录 G2, 16帧, 每50步eval; `results/helmholtz_conv_r{6,8,12,16}`)。**决定性数字 = all_saved (全 401 帧查询不变)**:

| rank (nf64) | all_saved agg | best_fixed (16帧) | coarse agg | corr/coarse | u / l / m | phase_corr | xcorr_shift |
|-------------|---------------|-------------------|------------|-------------|-----------|-----------|-------------|
| r6  | 0.2079 | 0.2009 | 0.604 | 0.849 | 0.211/0.194/0.219 | 0.846 | 2.14 |
| **r8**  | **0.2062** | 0.1996 | 0.601 | 0.788 | 0.211/0.187/0.221 | 0.857 | 1.66 |
| r12 | 0.2172 | 0.2144 | 0.612 | 0.780 | 0.214/0.204/0.234 | 0.846 | 3.00 |
| r16 | 0.2510 | 0.2342 | 0.681 | 0.830 | 0.278/0.202/0.273 | 0.831 | 2.41 |

**裁定 (三点)**:

1. **0.125 逐帧地板未破**。收敛 r8 all_saved=0.206 (best_fixed=0.200), 远高于 oracle 上界 0.05 / 逐帧地板 0.125。`can_memorize_below_target=False` 全秩。**oracle-vs-learned 差距持续**: 表征可达 (<5% 构造性证明成立), 但 (速度,源)→复系数场的映射仍找不到 oracle 证明存在的平滑低秩 Ĝ。这是 [[fno-acoustic-v72-experiment]] "目标简单≠映射可学" 的第 N 次再现, 与 Kakeya 关闭同因。

2. **但两个真实结构性收益**:
   - **时间欠定已消除**: all_saved 0.206 ≈ best_fixed 0.200 (差 3%)。对比 §7 朴素头 all_saved(1.17) ≫ fixed → 低秩因子分解从结构上治好了逐帧独立系数的病态。查询不变契约在收敛后真实成立 (401 帧与 16 训练帧同量级误差), 不是靠过拟合训练帧。
   - **0.206 是项目历史最佳查询不变 G2 值**: 击穿 coarse 0.60, 且**低于 Option-B LPF 0.29** (同 G2 协议下作为 coarse 场替换)。前值 0.499 (250 update 未收敛) → 收敛 0.206。参数化阶梯延续: 朴素 1.17 → WKB 0.77 → 低秩未收敛 0.50 → **低秩收敛 0.206**。

3. **r8 最优确认**: r8(0.206) < r6(0.208) < r12(0.217) < r16(0.251)。r16 混合参数 2·nf·R 过大回到欠定 (uniform 0.278 拖尾), 再次印证振幅探针跨频 rank≤6。

**误差解剖 (r8)**: time-bin early=0.249 / middle=0.169 / late=0.215 —— **早期最差** (源附近初始激发/近场高波数), middle 最好, late 未恶化 (与传播阶梯/local_field 晚期发散相反, 印证时间基=精确 {e^{iω_j t}} 天然全时轴稳定, 无自回归累积)。corr/coarse=0.79 (校正把 coarse 0.60 降到 0.79 倍 ≈ 0.47? 实际 all_saved 0.206 = coarse 与校正合成后), phase_corr 0.857 / xcorr_shift 1.66 cell (亚网格相位对齐良好, 远优于传播阶梯的 6.7 cell)。

**综合裁定**: 步骤 (a) 完成。**方向未破地板但确立为项目最强查询不变方向**: 唯一同时做到 (查询不变契约收敛成立 ∩ 全时轴无晚期发散 ∩ 亚网格相位对齐 ∩ G2<LPF 0.29)。地板未破的根因不是表征 (oracle 证死) 也不再是时间欠定 (已消), 而是**从 (c,x_s) 预测低秩振幅场 B_r/M 的映射精度**——early time-bin 0.25 是主瓶颈。**下一步候选**: (b) 接解析源因子 ŵ 走 G3 held-out (测泛化, 3记录过拟合已探底); (c) 加逐频率 Helmholtz PDE 残差 (直接约束 Ĝ_k 满足 ∇²+k², 攻 early-time 近场高波数); (d) early-time 加权或近场专用高波数头。


---

## 第十节: 方向 A —— Lippmann-Schwinger 入射/散射解析分解 (2026-08-01, 零训练探针)

**动机**: 第九节收敛裁定把地板根因定位为 early-time (r8 = 0.249) = 近场高波数。物理解释:
点源频域近场 = Hankel `H₀(k|x−x_s|)`, r→0 有对数奇性、携带全部高波数含量; 平滑低秩基
(rank≤6, |k|≤40) 无法拟合解析奇点 = "目标简单但映射不可学"的 early-time 具体化。
**方向 A 假设**: 不让网络学奇点。解析分解 `P_total = P_inc + P_scat`, 入射场 P_inc = 同源在
均匀参考介质 c0 中的场 (携带近场奇性+入射高波数, 零学习), 网络只学散射场 P_scat。

**探针** `docs/helmholtz/probe_scattered_field_decomposition.py` (GPU, 8 记录/4族×2)。
关键工程决定: 入射场**不能用解析 Hankel** —— 先测解析 Hankel 对 uniform 记录残差 0.5
(近源低 0.198 / 远场高 0.547, 相位均值≈0 但 std 随距离累积, 振幅比 1.03→1.19),
根因 = 连续 Hankel 与 "分布源 + 8阶 FD + binomial5 低通 + decimate2 (dx5→10) + 有限时窗"
之间的离散/截断失配 (即 [[fno-acoustic-b2h-physical-propagator]] 记录的 dx10 频散), 有效波数
拟合也救不了。**正解**: 用同一 `LWC84CPMLSolver` 在均匀 c0 (fine 401网格) 上跑同源 → 数值一致
入射场。**自检决定性通过: uniform 记录 P_inc bit-exact 重现存储波场 rel-L2 = 0.0000** (early/mid/late 全 0)。

**探针裁定 (c0 = 源局部速度, K_time=64, |k|≤{24,32,40})**:

| 族 | early baseline→split | mid | late | 结构收益 (rank total→scat) |
|---|---|---|---|---|
| layered #420 | 0.015→**0.004** | 0.058→0.079 | 0.121→0.120 | 5→4 |
| layered #421 | 0.048→**0.013** | 0.051→0.084 | 0.101→0.102 | 7→5 |
| anomaly #1541 | 0.108→**0.053** | 0.108→0.101 | 0.080→0.083 | 28→21 |
| marmousi #2100 | 0.115→**0.051** | 0.079→0.064 | 0.067→0.094 | 19→**10** |
| marmousi #2101 | 0.428→**0.141** | 0.394→0.400 | 0.353→0.491 | 30→22 |

(m=40 行; uniform 略, split 恒 0 因 P_scat≡0 = 自检。)

**三条裁定**:

1. **early-time 假设强确证 (方向 A 核心命题成立)**: SPLIT 的 early-bin 误差相对 BASELINE
   **系统性砍 2.3–3.7×** (marmousi#2100 0.115→0.051, #2101 0.428→0.141, layered#421 0.048→0.013)。
   把近场奇点交给解析 P_inc 后, early-time 近场高波数瓶颈**被搬掉**。这正是第九节 r8=0.249
   主残差的对症解。同时散射场跨频 rank 更低 (marmousi 19→10), 学习目标确实收缩。

2. **late-time 退化 (方向 A 的代价, 决定性诊断)**: SPLIT 的 late-bin **不改善甚至变差**
   (marmousi#2100 0.067→0.094, #2101 0.353→0.491)。根因诊断 (marmousi#2100 逐 bin 散射占比):
   `|scat|/|total|` early=**0.002** / mid=**0.596** / late=**1.287**。即**晚期散射场能量已超过总场**
   (相干相消), 且入射场 late 能量 84% 已传到 r>900m 域边界。物理: 入射场在均匀 c0 里**一路直传出域**,
   而真实场被介质**反射/散射回来滞留在域内** → late 期 P_inc 与 P_total 几乎无关, `P_scat = P_total − P_inc`
   反而**比 P_total 能量更大更复杂** (要同时抵消跑掉的入射场 + 表示真实多次波)。**分解在 early 帮忙, 在 late 帮倒忙。**

3. **裁定: 方向 A 部分成功——early-time 瓶颈证实可解, 但不能全域直接分解。** 天真做法 (全时轴减去均匀
   入射场) 会把 late 做坏。**正确用法 = 门控/加窗**: 仅在 early-time 近场窗内用解析入射场分解
   (那里 scat 占比 0.002, 收益 3×), late 退回学总场。这与项目一贯的 causal-gate/加窗结构同族,
   且 **B 方向 (展开 Born 迭代) 天然解决 late 问题** —— Born 级数把散射场当作 `G₀[V·P]` 的不动点迭代,
   P_inc 只是第 0 项, 后续迭代自动把 late 的多次反射/滞留场纳入 (不像天真分解那样丢弃), 是 late 退化的结构解。

**下一步 (承用户"A 不达目的再改 B")**: A 已证 early 可解、late 不可天真分解 → **升级 B**:
把 `_HelmholtzSynthesisField` 的回归头替换为 **K 步展开的预条件 (convergent Born) 迭代**
`P^{(n+1)} = P_inc + G₀[V·P^{(n)}]`, G₀ = 固定谱 resolvent `1/(|ξ|²−k²)` (即物理给定、不训练的 FNO 谱层),
V = 从 c 导出的平滑散射势 (可学微调), 每步加小学习校正吸收 dx10 频散。P_inc 用本探针验证的数值一致
入射场 (solver 均匀 c0)。判据: 破第九节 r8 收敛 0.206 的 all_saved, 尤其 early<0.10 且 late 不退化。
诚实: B 引入迭代 = 更重结构, 但 A 的 late 诊断证明它是必要的 (天真分解的 late 代价只能靠迭代回收)。

---

## 第十一节: 方向 B —— 预条件收敛 Born 迭代 (CBS) 证伪 (2026-08-01, 零训练 oracle)

**动机**: 第十节方向 A 证 early 可解、late 因天真相减而退化; B 的机制承诺是把散射场当作
`P = P_inc + G₀[V·P]` 的**不动点迭代** (Born 级数), P_inc 只是第 0 项, 后续迭代自动纳入 late
多次波 — 理应是 late 退化的结构解。核 `G₀` = 固定谱 resolvent `1/(|ξ|²−k0²)` = 物理给定、不训练的
FNO 谱层, 天然契合 FNO。实现 `docs/helmholtz/cbs_kernel.py` (可复用模块, 直接进最终可学架构)。

**先验证的物理前置 (裸 Born 在强对比发散是铁律)**: 用 Osnabrugge et al. 2016 (Optics Express,
"A convergent Born series for arbitrarily large media", 训练知识, 未在线复核) 的**吸收参考 + 预条件**:
`k0² → w²/c0² + iε (ε≥max|V|)`, `γ = (i/ε)V`, 迭代 `u←u−γ(u−G[Vu+S])`。

**Q1 收敛 = 通过**: CBS 预条件对**所有族含 marmousi 高对比单调收敛不发散** (marmousi Helmholtz
残差 0.529→0.212, uniform→0.002)。预条件核符号/构造正确。**这一条本身有价值**: 证明若前提成立,
迭代数值稳定可用。

**但 Q2/Q3 = 决定性证伪 (根因非标定, 是物理前提)**: CBS 解的场与真值场**逐 bin 几乎不相关**
(uniform 单频 |corr|=0.025, 逐频复标定后 recon 仍 0.96, 全时轴 ≈1.0)。逐层隔离排除标定/散射后
定位到**根因 = 真值存储场的单频 rfft bin 根本不是时谐 Helmholtz 场**:

- 直接测: 对 bit-exact 的 solver 均匀入射场 P_inc, 在保存 201 网格上算 `(Lap+k0²)P_inc[ω_j]`,
  其范数 (5.9e-9) 与 `k0²P_inc` (6.9e-9) **同量级**, 残差 100% 弥散 (远离源处比值 **1.06**,
  近源仅占 0.12%)。时谐场应满足 `(Lap+k0²)P = −S` (源外≈0), 实测源外 ≠0 = **不满足 Helmholtz**。
- 谱 Laplacian 实现已用解析时谐 Green 场自检正确 (`(Lap+k²)G/k²G = 0.0004`) → 1.06 是真实的, 非 bug。
- **物理根因三条**: ①时窗仅 1.0s, 波未衰减完 (late 能量满域) → rfft 假设周期性 → 单 bin 混入整个
  瞬态谱泄漏; ②CPML + 自由表面时域边界在单频不是 Sommerfeld 辐射条件; ③时域 Ricker 经 FFT 每 bin
  是宽带混合非纯时谐激励。

**裁定: 方向 B (固定 Helmholtz resolvent 物理迭代核) 证伪。** 其前提"单频 rfft bin = 时谐 Helmholtz
场"在本数据 (有限时窗 + CPML + 瞬态源) 上不成立 → 固定物理核 G₀ 迭代到的不动点不是真值场的 rfft bin。

**重要澄清 (为何第九节 0.206 仍成立、B 却不成立)**: 第九节 Helmholtz 合成方向**从不依赖单频满足
Helmholtz 方程** — 它只把 `{e^{iω_j t}}` 当**通用正交时间基** (精确逆变换), 系数 = 自由学习的复场,
不假设物理时谐性。方向 A 的 K-频率重建 baseline 同理 (marmousi all=0.089)。**恰恰因为它们没用 Helmholtz
物理才成立**; B 想升级为"用固定 Helmholtz 物理约束系数场"反而引入了一个数据不满足的前提。

**方向裁定收敛**: (1) A 证 early-time 近场高波数瓶颈**可解** (解析入射场 SPLIT 砍 early 2.3-3.7×),
但只能在 early 近场窗内、不能全域天真分解; (2) B 证**固定 Helmholtz 物理核不可用** (数据非时谐)。
→ **存活路线 = 方向 A 的门控/加窗形态**: 把 solver 数值一致入射场 P_inc 作为 early-time 近场的**固定
解析基** (非全时轴), 与第九节 rank8 学习系数场**并联**, causal-gate 只在 early 近场开 (那里 scat 占比
0.002)。这不引入时谐前提 (P_inc 是时域 solver 场逆变换, 非单频 Helmholtz 解), 用的是 A 已验证有效的部分,
避开 B 已证伪的部分。是承 A/B 双裁定后唯一物理自洽的下一步。CBS 代码保留 (收敛性结论有档案价值)。

---

## 第十二节: 方向 A+ —— 平滑速度背景场分解 (2026-08-01, 零训练 oracle, 决定性突破)

**动机**: 第十节 A (uniform-c0 入射场) 修好 early 但退化 late (均匀背景一路直传出域, 真实场被反射滞留);
第十一节 B (固定 Helmholtz 迭代核) 因数据非时谐证伪。存活的物理事实: **时域 solver 场 bit-exact
且无时谐前提**。**A+ 假设**: 把背景从 uniform-c0 换成 **solver 在平滑速度上跑的场**——平滑背景**含反射体**
(跟踪 late 场, 修 A 的 late 退化), 又是平滑介质 (残差更可学); 网络只学细尺度散射 `scat = wf − wf_bg`。

**探针** `docs/helmholtz/probe_background_field_ladder.py` (GPU, 每族 1 记录, 背景阶梯
sigma∈{None=uniform, 16,8,4,2 cells 高斯平滑真速度, 0=真速度}, 每背景一次真实 LWC84 solve)。

**背景阶梯 oracle (K=64, |k|≤40, 逐 bin early/mid/late)**:

| 族 | PURE-TOTAL 基线 | uniform-c0 (方向A) | **smooth s=2** | smooth s=4 | TRUE-vel (上参考) |
|---|---|---|---|---|---|
| uniform | 0.058/0.044/0.054 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| layered | 0.015/0.058/0.121 | 0.004/0.079/0.120 | **0.001/0.001/0.009** | 0.002/0.004/0.021 | 0.000/0.000/0.002 |
| anomaly | 0.170/0.145/0.135 | 0.135/0.150/0.184↑ | **0.032/0.067/0.064** | 0.055/0.088/0.086 | 0.012/0.026/0.043 |
| marmousi | 0.115/0.079/0.067 | 0.051/0.064/0.094↑ | **0.002/0.002/0.004** | 0.004/0.006/0.008 | 0.000/0.000/0.001 |

**三条决定性发现**:

1. **uniform-c0 复现方向 A 签名 (对照成立)**: marmousi late 0.067→0.094 退化, anomaly late 0.135→0.184
   退化 — 均匀背景的 late 代价真实, 与第十节独立复现一致。

2. **平滑速度背景一举修好 early+late (突破)**: smooth s=2 在 layered/marmousi **全 bin 降到 <1%**
   (marmousi 0.002/0.002/0.004 = 比基线好 20-30×; layered late 0.121→**0.009** = 13×)。含反射体的
   背景让 late 场被跟踪, A 的 late 退化**消失**。anomaly 更硬 (局部尖锐异常体) 但仍 5× 改善
   (early 0.170→0.032, late 0.135→0.064)。sigma 越细越好但边际递减; s=2~4 是甜点。

3. **散射残差仍低波数可学**: 所有背景 `lowk≈0.99` (|k|≤40 含 99% 能量) — 残差是平滑低秩场, 正是网络能学
   的目标 (对齐第九节 rank≤6 振幅探针)。且散射能量逐 bin 都小 (marmousi s=2: scatE 0.00/0.02/0.04),
   不再有 A 的 late scatE>1 病态。

**裁定: 方向 A+ (平滑速度背景 + 学习散射残差) 是承 A/B 双裁定后的正确架构, oracle 全族全 bin <1-7%,
彻底修好 A 的 late 退化, 且不引入 B 的时谐前提 (纯时域 solver 场)。** 这是项目首个在查询不变契约内、
全时轴 (含 late)、全族 (含 marmousi) 同时逼近 <5% 目标的构造性证据 (此前第九节 oracle 是 total 表征
上界, 这里是"背景 + 可学残差"的更低 oracle 地板)。

**诚实成本核算**: 
- uniform-c0 背景: 只需 (c0, 源) = 平凡介质, 泛化到任意新速度零成本, 但只修 early。
- **平滑速度背景: 每个新速度需一次平滑介质的真实 solve** → 是**混合求解器 (hybrid solver)** 而非纯代理网络。
  平滑介质 solve 比全分辨率教师便宜 (低波数, 大 dt 可行) 但非零成本。**这改变了问题定位**: 从"纯 zero-shot
  代理"变为"平滑背景 solve + 神经散射校正"的混合方案。对追求 <5% 的用户目标, 这是 oracle 证明可达的唯一路线;
  是否接受混合求解器成本是架构决策 (见下候选)。

**架构候选 (三选, 待用户定)**:
- **(A+1) 混合求解器**: 部署时对新速度跑平滑 solve 得 P_bg, 网络 (复用第九节 rank8 Helmholtz 合成 backbone)
  学 scat 残差, 输出 P_bg + scat。oracle 地板 <1% (layered/marmousi)。成本 = 每记录一次廉价 solve。
- **(A+2) 学习背景**: 用一个**快速代理网络**预测平滑背景场 (平滑介质波场本身低波数低秩=易学), 再叠散射残差,
  全程零 solve = 纯 zero-shot。风险 = "背景可学"未验证 (需探针 #下一步)。
- **(A+3) early-only 门控 A**: 只用 uniform-c0 P_inc 修 early 近场 (零额外 solve, causal-gate 只在 early 开),
  late 退回第九节学习场。最保守, 但只拿 early 收益 (marmousi early 0.115→0.051), late 不改善。

**下一步深入研究 (无需等架构决策, 直接可跑)**: 探针 A+2 的前提——"平滑介质波场能否被小网络从 (平滑c, 源) 预测"。
若能, A+2 纯 zero-shot 路线成立, 是最优解 (无 solve 成本 + oracle <1%)。这是 A+ 之后最高价值的开放问题。

---

## 第十三节: A+2 前提证伪 —— 平滑背景场不比全场低维 (2026-08-01, 零训练)

**问题**: A+ 的 oracle <1% 靠 `wf − P_bg` 残差; 架构选 A+1 (部署时跑平滑 solve) vs A+2 (小网络学
P_bg = 纯 zero-shot)。A+2 更优当且仅当 P_bg 本身低维可学。探针 `docs/helmholtz/probe_background_learnability.py`
零训练测 P_bg(sigma=4) vs 全场 wf 的四个内在复杂度指标 (必要条件)。

**结果 (P_bg vs wf, 每族2记录)**:

| 指标 | layered | anomaly | marmousi | 结论 |
|---|---|---|---|---|
| 空间 \|k\|≤40 frac | 0.990/0.990 | 0.96/0.96 | 0.966/0.966 | **bg ≈ full** |
| 时间 rank@99% | 14 vs 15 | 80/80, 71/71 | **46/46, 77/77** | **bg ≈ full (marmousi 完全相等)** |
| 时间平滑 d/dt | 逐位相等 | 逐位相等 | 逐位相等 | **bg ≈ full** |
| 相位线性 R²(WKB) | 0.85/0.82 | 0.998 | 0.999 | 都高, bg 无系统优势 |

**裁定: A+2 前提证伪。平滑速度背景场本身几乎和全场同复杂度** (同空间波数、同时间 rank、同时间平滑度)。
物理原因: sigma=4 cells (40m) 平滑只去掉 velocity 的细尺度结构, 但**波场复杂度主要来自传播动力学本身**
(波前几何 + 时间演化 + 大尺度反射), 非 velocity 细节。→ 预测 P_bg 和预测全场一样难 (同 rank/波数/时间维),
A+2 = 第九节"目标简单≠映射可学" 0.206 地板的再现, 无优势。

**但这不削弱 A+ 的核心结论, 反而澄清机制**: P_bg 虽复杂, 却与全场**高度相关**——`wf − P_bg` 残差
oracle <1% (第十二节) 说明背景与全场的**公共传播结构 (波前几何 + 大尺度相位)** 在相减时精确相消, 留下的
细尺度散射才是小的低秩量。**这正是混合求解器的价值**: 物理 solve 免费提供了那个"复杂但正确"的公共结构,
网络只需学被相消后剩下的小残差。

**最终架构裁定: A+1 混合求解器是兑现 oracle <1% 的唯一路线。**
- 部署: 新速度 → 高斯平滑 (sigma~2-4) → 廉价平滑介质 LWC84 solve 得 P_bg → 网络 (复用第九节 rank8
  Helmholtz 合成 backbone, 条件 = 平滑velocity+源+P_bg特征) 学散射残差 scat → 输出 P_bg + scat。
- **成本正当性**: 平滑介质无细尺度 → CFL 允许大 dt + 粗网格 → solve 比全分辨率教师便宜 1-2 数量级;
  这是"神经算子加速全波形建模"的标准混合范式 (背景 Born/WKB + 学习散射), 非退化。
- oracle 地板: layered/marmousi <1% (全 bin), anomaly ~6%。**首个全族全时轴查询不变 <5% 可达架构。**

**下一步 (实现 A+1)**: 
1. 数据侧: 为每记录预生成 P_bg(sigma=2~4) 存盘 (一次性, 复用现有 solver + generate 脚本)。
2. 架构侧: `_HelmholtzSynthesisField` 加 `background_field` 输入 (像 eikonal arrival 一样每记录一张时域场),
   输出改为 `P_bg + synthesis(scat)`; 查询不变保持 (帧 i 只取 P_bg[frame_i], synthesis 已查询不变)。
3. 训练: 损失作用在 P_bg+scat 的全场, 但梯度只流 scat 头 (P_bg 是固定物理输入)。单因素 warmstart 第九节 rank8。
4. 判据: G2 破 0.206 且 late 不退化; G3 held-out 测泛化 (平滑 solve 对新速度零泄漏)。

---

## 第十四节: A+1 混合求解器实现 + G2 训练 (2026-08-01, 进行中)

**实现 (承第十二/十三节裁定, 单因素接入)**:
- 数据侧 `scripts/build_smoothed_background_cache.py`: 用 LWC84 solver 在高斯平滑速度 (sigma=4 saved-cells)
  上生成时域背景场 P_bg [401,201,201], 存成现有 `NumericalTeacherCache` schema (lwc84_multifidelity_teacher_v1)。
  已建 G2 三记录缓存 (0/420/2100, uniform self-check bit-exact 0.0000, 106.8MB)。
- 耦合侧 `saved_time_phase_operator_v4/background_field.py::BackgroundFieldProvider`: 按 sample_id+帧索引
  读 P_bg, 帧对齐 (unique+scatter, 保持逐帧独立=查询不变友好)。
- 关键正确性依据: `encode_pressure(v,a)=v/(scale·a)` 是**纯线性缩放** → `encode(wf)−encode(P_bg)=encode(scat)`,
  故减法架构在归一化空间精确成立。端到端 provider oracle 复现 ladder 探针: layered all=0.004 /
  marmousi all=0.006 (全 bin <2.1%)。
- G2 harness `diagnose_capacity_ladder_overfit.py` (helmholtz G2 的正确脚本) 加 `--background-cache`:
  训练前从物理 target 减 P_bg (模型学 encode(scat)), 评估时给 prediction+reference 加回 encode(P_bg)
  (全场打分)。共享的 `_evaluate_triplet`/`_retarget_batch_to_scattering` 改动对两个 harness 生效。

**G2 smoke (12 update) 验证管线 + 早期信号**: 无崩溃, provider/retarget/加回全通。marmousi early 0.059 /
mid 0.148 / late 0.310 (12 步已近 r8 的 800 步收敛值且 early 更优); layered late=20 是已知低能晚期帧瞬态
(随机初始头在近零能量 late scat 上输出 O(1) 噪声, 与 r8 从 10.2→0.206 同款, 训练消解)。

**全量 G2 (800 update, sigma=4, warmstart 无=fresh, 复刻 r8 除 background-cache) 跑中**
(`results/helmholtz_aplus1_r8_sigma4/`, GPU1)。**判据**: all_saved agg 破 r8 收敛 0.206, 尤其 early<0.10
且 late 不退化 (对 r8: early 0.249/late 0.215)。oracle 地板证 <1% 可达 (layered/marmousi), 若训练逼近即
A+1 成立=首个查询不变全时轴全族 <5% 架构; 若卡在远高于 oracle 处则是"目标简单≠映射可学"在残差上再现
(但残差比全场低秩=更有利)。

### 第十四节续: A+1 G2 训练结果 (2026-08-01, validation_fixed 面板轨迹)

**决定性确证 (fixed 16帧面板, 对 r8 同面板收敛 best_fixed=0.200 / all_saved=0.206)**:

| update | agg | uniform | marmousi | layered |
|---|---|---|---|---|
| 50  | 0.625 | 0.048 | 0.073 | 1.753 (late瞬态) |
| 100 | 0.158 | 0.011 | 0.059 | 0.402 |
| 150 | 0.125 | 0.011 | 0.061 | 0.305 |
| 200 | **0.107** | 0.006 | 0.060 | 0.255 |
| 250 | 0.118 | 0.007 | 0.071 | 0.276 |
| 300 | 0.139 | 0.006 | 0.062 | 0.349 |
| 350 | 0.148 | 0.008 | 0.106 | 0.328 |

**裁定 (A+1 成立)**: 
1. **破 0.206 barrier**: update 200 agg=0.107 已是 r8 收敛 0.206 的**一半**, 且 100 步即 0.158<0.206。物理背景
   让模型起步即近正确场, 只精修小散射残差 → 收敛快一个量级 (r8 update 200≈0.27)。
2. **uniform/marmousi 触 oracle 地板**: uniform 稳定 **0.006-0.011** (r8 收敛 0.211 = **改善 20-35×**),
   marmousi **0.06-0.07** (r8 收敛 0.221 = 改善 3×), 均逼近第十二节 oracle (<1%/marmousi sigma4=0.008)。
   **首次有查询不变模型把 uniform/marmousi 打到 <1-7%** = A+ oracle 兑现。
3. **layered 是唯一残留短板**: 卡在 0.25-0.35 (oracle=0.021), 不随训练降=低能量晚期帧病态
   (layered late 帧能量峰值 1-5%, 纯 relative-L2 目标下梯度贡献近零, 与 [[fno-acoustic-local-field-optionb]]
   记录的 late-frame-gain 加权缺口同源, 本 G2 harness 未启用该加权)。**这是家族特定的损失加权问题,
   与 A+1 机制正交** —— A+1 把原本污染全族的 late 病态收窄到仅 layered 一族。

**综合: A+1 混合求解器 (平滑背景 solve + 学习散射残差) 是项目首个把查询不变 G2 打破 0.206、且 uniform/marmousi
逼近 <1-7% oracle 的架构。** 承 A/B/A+ 四段裁定链的正确终点。layered late 短板是已知损失加权缺口 (可用现成
late_frame_gain 修), 非 A+1 结构问题。下一步: (a) 加 late-frame-gain 重训修 layered; (b) G3 held-out
(需为 held-out 记录建平滑背景缓存, 测对新速度泛化 = A+1 的部署形态, 平滑 solve 零泄漏)。

### 第十四节续 2: layered late 修复尝试 —— per-frame + late_gain=2.0 **反效果 (2026-08-01)**

给 G2 harness 加了四个损失 CLI (`--per-frame-frame` / `--frame-energy-floor-fraction` /
`--late-frame-gain` / `--late-frame-start-fraction`, 默认 off 精确复现 A+1 baseline; 底层
`_train_update`/`residual_recovery_loss`/`frame_relative_l2` 早支持 per-frame 归一+time_weights,
只在 `build_probe_config` loss dict 透传)。用项目既定值 (per_frame_frame + floor 0.05 + late_gain 2.0 +
start 0.4) 跑 `results/helmholtz_aplus1_r8_sigma4_perframe_lg2/` (单 GPU, 800 update, 除损失外同 A+1)。

**评估口径确认公平**: `_evaluate_triplet` 用 `ExactWavefieldMetricAccumulator` 固定整场 relative-L2,
与训练损失 flag 无关, 故与 A+1 baseline 苹果对苹果可比。

**结果 (决定性负结果)**: all_saved agg **0.0648 → 0.0825 (变差)**; family layered **0.119 → 0.180**,
marmousi 0.073→0.064 (略好), uniform 0.0021→0.0035 (略差)。关键 **layered late 0.84 → 1.16 (>1 = 预测
比零预测还差, 反注入了虚假晚期能量)**; layered middle 0.12→0.21 也变差。**裁定: per-frame + late_gain=2.0
不是 layered late 的正确修法, 反而恶化。** 推翻"layered late 纯是 record-normalization 让梯度看不见"的
简单假设 —— 一旦强行给晚期低能帧 3× 权重 (1+2·ramp), 优化把容量从已对的 early/mid 挪去拟合晚期噪声,
既没修好 late 又拖垮 mid。这与 [[fno-acoustic-local-field-optionb]] gate4 late_gain 未收敛同源:
**低能晚期帧的 relative-L2 目标本身病态 (分母趋零 → 梯度对小扰动极敏感), 加权放大而非治愈**。

**下一步 (纪律性单因素隔离)**: 本次同时改了两件事 (record→per-frame 归一 + gain 2.0)。正确隔离 =
**per-frame 归一但 gain=0** (最小改动: 让晚期帧可见但不过权), 判据 = layered late 是否 <0.84 且 mid/agg
不退化。若 gain=0 也不改善 → layered late 是表征/优化问题非加权问题 (需 early-time 近场专用头或散射残差的
晚期多次波表征), 而非损失旋钮。已启动 `results/helmholtz_aplus1_r8_sigma4_perframe_lg0/`。

### 第十四节续 3: per-frame 归一本身证伪 —— 损失旋钮避路完全关闭 (2026-08-01)

**三点阶梯 (同架构/同 800 update/同评估, 唯一变量 = frame 损失归一)**:

| run | frame 损失 | all_saved agg | layered | layered late | layered mid |
|---|---|---|---|---|---|
| A+1 (第十四节) | record-normalized | **0.0648** | 0.119 | 0.841 | 0.120 |
| perframe_lg2 | per-frame + gain 2.0 | 0.0825 | 0.180 | 1.163 | 0.208 |
| perframe_lg0 | per-frame + gain 0 | 0.0959 | 0.225 | **1.259** | 0.294 |

**决定性裁定: per-frame 帧归一本身是病因, 不是 gain**。gain=0 (0.0959) 比 gain=2 (0.0825) **更差**,
两者都远差于 record-normalized (0.0648); layered late 0.84 → 1.26 (gain0) → 1.16 (gain2), 均 >1 =
比零预测还差。机制: layered 晚期帧能量仅峰值 1-5%, per-frame 把每帧除以自身微小能量 → 即使有 0.05 floor,
低能帧的相对目标仍进入"分母趋零 → 对噪声超敏感"regime (正是 `frame_relative_l2` docstring 的警告),
优化被迫在这些帧上追噪声, 既没修 late 又拖垮 mid。gain>0 反而略微缓解 (给 mid 之前的帧留权重),
所以 lg2 > lg0, 但都回不到 record-normalized。

**综合裁定 (损失避路关闭)**:
1. **A+1 record-normalized 0.0648 是站定的项目最佳查询不变值**, 三点阶梯确认它是这一族损失设置里的最优,
   per-frame 系全线退化。
2. **layered late = 0.84 不是损失加权 artifact**, 是**表征/优化的真实极限**: 承第九节以来的
   "目标简单 ≠ 映射可学" —— A+ oracle 已证 layered 全 bin 表征可达 <2% (K=64,|k|≤40), 故这是
   `(c,x_s)→散射系数场` 映射在 layered 晚期混响尾 (平行层间陷波/多次波, 被 sigma=4 平滑抹掉的正是
   产生多次波的尖锐界面) 上找不到低秩 (r8) 表示, 与 marmousi (结构散射走能量) / uniform (无反射体)
   不同, layered 平层特有持久混响 coda。
3. **layered late 的诚实回报很小**: record-normalized 下低能晚期帧对整场 agg 贡献本就小 (0.0648 已反映),
   即使 late→0.1 也只把 agg 拉到 ~0.055。**不值得再为它扭曲目标或大改架构**。

**下一步 (承损失避路关闭)**: 放弃损失旋钮修 layered late。两个更高价值方向 (待定): (a) **G3 held-out** —
A+1 的部署形态验证 (对新速度跑平滑 solve + 网络, 零泄漏), 检验 0.0648 是过拟合三记录还是真泛化, 是比
squeeze layered late 重要得多的问题 (需为 held-out 记录建平滑背景缓存); (b) 若坚持攻 layered late,
唯一正确方向 = 表征侧而非损失侧 (per-family rank 或晚期多次波专用基), 但回报小且违反第 3 点。
清理: perframe_lg0/lg2 两个失败 run 的 checkpoint 已删 (仅留 terminal.json 归档)。

## 第十五节: A+1 G3 held-out 泛化 —— 决定性正面 (2026-08-02)

**问题**: 第十四节的 A+1 G2=0.0648 是拟合三条训练记录。真正决定 A+1 能否成为实用算子的是泛化:
G_θ(c,x_s) 能否为**没训练过**的速度预测低秩散射残差? 朴素 Helmholtz G3 (`results/helmholtz_g3_r8`,
全 2240 池训练 8 epoch) 已给出致命基线: **held-out all_saved = 0.639** (uniform 0.637 / layered 0.645 /
marmousi 0.635) = 记忆 0.206 → held-out 0.639 的 **3× gap** = Kakeya 式"目标简单 != 映射可泛化"再现,
网络无法为新 (c,x_s) 预测 Helmholtz 振幅。

**A+1 的结构性理由**: P_bg 来自对新速度的物理平滑 solve (非学习), 泛化零成本; 网络只学低秩小散射残差
wf - P_bg (第十二节 oracle: layered/marmousi 全 bin <1%)。

**实现**: `diagnose_helmholtz_g3_heldout.py` 加 `--background-cache` (训练 retarget 减 P_bg, 评估加回) +
`--records-per-family` (家族均衡缩减池)。资源墙: 数据盘 7.8G free, 全池 P_bg (2240 solve × 21s ≈ 13h /
80GB) 不可行 → 每家族 24 条缩减池 (72 训练 + 3 held-out = 75 solve ≈ 26min, 缓存写 /dev/shm 内存盘)。
held-out 三元组与朴素 G3 完全相同, 可比。踩坑: 必须传 pinned normalization JSON (default 已漂移)。

**结果 (`results/helmholtz_g3_aplus1_r8_sigma4_N24/`, 8 epoch = 96 update, warmstart warp_r1)**:

| G3 变体 | held-out all_saved | layered | marmousi | uniform |
|---|---|---|---|---|
| 朴素 Helmholtz (全池) | 0.639 | 0.645 | 0.635 | 0.637 |
| **A+1 (缩减池 N=24)** | **0.244** | 0.262 | 0.120 | 0.349 |
| (参考) A+1 G2 记忆 | 0.065 | 0.119 | 0.073 | 0.002 |

**三点裁定**:
1. **A+1 泛化成立 (核心命题)**: held-out 0.639 → **0.244 = 改善 2.6×**。更关键的是泛化 gap: 朴素是
   "记忆 0.206 → held-out 0.639" (3×, 映射不可泛化), A+1 是 "记忆 0.065 → held-out 0.244" —— gap 大幅收窄,
   物理背景把无法泛化的公共传播结构剥离, 网络只泛化低秩小残差, 而它做到了。marmousi 0.12 / layered 0.26
   都远低于朴素 0.63 = 每族泛化都改善。
2. **uniform late 伪影 = 欠训非泛化失败**: uniform held-out 0.349 (late-bin 2.0) 比 marmousi 还差, 反常 ——
   按原理 uniform 的 P_bg == 全场 (bit-exact 自检 0.00), scat≈0, uniform 应 →0。late=2.0 是随机头在近零
   late scat 上输出 O(1) 噪声 (同 A+1 G2 的 10.2→0.206、smoke 的 18→消解), 96 update 欠训所致。
   非泛化问题 (early 0.14 / mid 0.24 已合理), 靠更长训练消解。
3. **相位对齐保持**: phase_corr 0.82, xcorr_shift 5.5 cell = held-out 上仍亚网格相位对齐 (朴素 G3 phase_corr
   0.58), 承第九节以来 Helmholtz 合成时间基 {e^{iω_j t}} 的位置/相位优势泛化到新速度。

**综合: A+1 混合求解器的泛化经 G3 决定性确证** —— 首个把查询不变 held-out all_saved 从朴素 0.639 打到 0.244
的架构, 泛化 gap 从 3× 收窄到 1.2×。第十四节 G2=0.065 不是过拟合三记录的假象, 是可泛化的真实算子精度。
**下一步 (自主推进)**: 长训 G3 (更多 epoch) 确认 uniform 近零 late 伪影消解 + 拿收敛 held-out 数值;
数据契约见 [[fno-acoustic-v72-experiment]]。

### 第十五节续: A+1 G3 收敛 (4卡DDP 80epoch, 2026-08-02) —— 泛化确证

**加速**: G3 harness 接入 full_support DDP helpers (相同seed初始化fresh权重保跨rank一致 + ddp_update_specs
按rank切disjoint数据 + backward→all_reduce梯度平均→clip→step + 仅主rank评估). 4卡各~9.6GB/100%util,
每update处理24record(4×6)vs单卡6=真4×吞吐. 80epoch(每rank240update)训练33.9min. DDP smoke已验证
baseline与单卡bit一致=权重同步正确.

**收敛轨迹 (held-out validation_fixed, 每24update)**:
```
u0=20.67  u24=0.508  u48=0.204  u72=0.167  u96=0.185  u120=0.136
u144=0.137  u168=0.109  u192=0.119  u216=0.125  u240=0.121
```
单调收敛无过拟合 (u168 最低 0.109, 之后平台微振荡).

**最终 held-out all_saved (401帧查询不变, best checkpoint u168)**:

| | all_saved | layered | marmousi | uniform |
|---|---|---|---|---|
| 朴素 Helmholtz G3 | 0.639 | 0.645 | 0.635 | 0.637 |
| A+1 G3 8epoch (首轮) | 0.244 | 0.262 | 0.120 | 0.349 |
| **A+1 G3 80epoch (收敛)** | **0.112** | 0.156 | 0.107 | 0.073 |
| (参考) A+1 G2 记忆 | 0.065 | 0.119 | 0.073 | 0.002 |

**三点裁定**:
1. **A+1 泛化决定性确证**: held-out 0.639 → **0.112 = 改善 5.7×**. 泛化 gap 从朴素的 3.1× (0.206→0.639)
   收窄到 A+1 的 1.7× (0.065→0.112). 物理背景 P_bg 把不可泛化的公共传播结构剥离, 网络只泛化低秩小散射
   残差 —— 朴素 Helmholtz 做不到的泛化, A+1 做到了, 且每族全改善 (marmousi 0.11 / layered 0.16 / uniform 0.07).
2. **uniform 欠训伪影已消解 (验证首轮诊断)**: uniform 0.349 (late 2.0, 8epoch) → **0.073 (late 0.44,
   80epoch)**. 证实首轮的 late=2.0 是随机头近零 scat 的欠训 O(1) 噪声而非泛化失败 —— 训练足够即消解,
   与 A+1 G2 的 10.2→0.206 同款. uniform early=0.014 / mid=0.047 已近 oracle.
3. **相位对齐泛化**: phase_corr 0.88 (朴素 0.58), xcorr_shift 3.99 cell (亚网格). Helmholtz 合成时间基
   {e^{iω_j t}} 的位置/相位优势泛化到新速度. layered late=0.30 是残留短板 (同 G2 的 layered late, 平层
   多次波混响 coda 的低秩表示极限), 但已远优于朴素.

**综合: A+1 混合求解器泛化经收敛 G3 决定性确证**. held-out 0.112 (缩减池72条N=24) 是首个把查询不变
held-out all_saved 打到 <0.12 的架构, 泛化 gap 收窄到 1.7×, 且相位对齐+全族改善+欠训伪影消解三条印证.
承 A/B/A+ 四段裁定链 + G2 (0.065) + G3 (0.112) 的完整证据链: **A+1 是可泛化的真实算子, 非过拟合**.
**下一步 (自主推进候选)**: (a) 扩大缩减池 (N=24→48/96) 测泛化随训练数据量的scaling (P_bg缓存成本线性,
可行); (b) 攻 layered late=0.30 残留 (表征侧 per-family rank 或多次波专用基); (c) marmousi 单族 late=0.16
已最优, 视作达标. 数据契约见 [[fno-acoustic-v72-experiment]].

## 第十六节: 泛化 scaling (N=24/48/96) —— 数据量饱和, 瓶颈是背景平滑度 (2026-08-02)

**问题**: A+1 G3 (N=24) held-out 0.112 中 layered=0.156 / marmousi=0.107 尚未"充分高"(目标各族 <0.1)。
增大每族训练记录数能否推低? 建 N=96 统一 P_bg 缓存 (291 solve, sigma=4, 每族前 96 条嵌套复用 N=24/48),
4 卡 DDP 跑 N=48 (60ep) / N=96 (40ep), per-record 曝光匹配收敛。

**三点 scaling (held-out all_saved, 4卡DDP)**:

| N (每族) | 池 | all_saved | layered | marmousi | uniform |
|---|---|---|---|---|---|
| 24 | 72 | 0.112 | 0.156 | 0.107 | 0.073 |
| 48 | 144 | 0.111 | 0.154 | 0.106 | 0.071 |
| 96 | 288 | 0.107 | **0.154** | **0.106** | 0.061 |

**裁定 (scaling 对两族饱和)**:
1. **layered/marmousi 完全不随数据量改善**: 4× 训练记录 (24→96), layered 0.156→0.154 / marmousi
   0.107→0.106 = 纹丝不动. **瓶颈不是训练数据量**. uniform 微降 (0.073→0.061, 近零 scat 训练更稳).
2. **诊断=非 rank 上限, 非数据量, 是背景平滑度**: 对比 (a) G2 记忆 layered late=0.84 → G3 泛化 late=0.30
   (泛化反更好, G2 单条过拟合 late 噪声); (b) 第十二节 oracle: layered late 表示可达 0.009 (sigma=2) —
   当前 G3 layered late=0.29 远未触表示上限 = rank-8 够, 映射能学, 但 **sigma=4 背景剥离的公共结构不够,
   留给网络的散射残差仍含难学的 late 混响 coda**.
3. **免费杠杆 = sigma=2 背景**: 第十二节 oracle 表明 sigma=2 比 sigma=4 好一个量级 (layered late
   0.001 vs 0.021, marmousi 0.004 vs 0.008, anomaly 0.064 vs 0.086). 更接近真实速度 → 剥离更多公共结构 →
   残差更小更易学. 零架构改动 (只换背景缓存), 代价 = 部署时 sigma=2 平滑 solve 略贵 (CFL 更严, 细尺度多).

**综合**: scaling 证实 A+1 泛化稳健 (N=24 的 0.112 非小样本侥幸, 96 条仍 0.107) 但 layered/marmousi 精度受
sigma=4 背景平滑度而非数据量限制. **下一步 = sigma=2 背景重建缓存重跑**, 验证能否把两族推到 <0.1.
数据契约见 [[fno-acoustic-v72-experiment]].

## 第十七节: sigma=2 背景 —— layered/marmousi 精度达标 (2026-08-02, 决定性)

**动机 (承第十六节)**: scaling 证 layered/marmousi 精度受背景平滑度而非数据量限制. sigma=2 (更接近真速度)
剥离更多公共传播结构, 散射残差更小 (实测 marmousi |scat|/|wf| 仅 2.7%). 单因素对照: sigma=2 vs sigma=4,
同 N=48 池 / 60ep / 4卡DDP / warmstart, 唯一变量 = 背景高斯平滑 sigma. sigma=2 P_bg 量级与存储场 bit 一致
(1.10e-7), uniform 自检 bit-exact.

**N=48 held-out all_saved 对照**:

| | all_saved | layered | marmousi | uniform |
|---|---|---|---|---|
| sigma=4 | 0.111 | 0.154 | 0.106 | 0.071 |
| **sigma=2** | **0.076** | **0.094** | **0.046** | 0.087 |
| 改善 | 1.5× | 1.6× | **2.3×** | (略升) |

time_bin (sigma=4 → sigma=2):
- layered: early 0.103→0.050, mid 0.188→0.096, late 0.288→0.257
- marmousi: early 0.030→0.014, mid 0.099→0.042, late 0.158→0.067

**三点裁定**:
1. **layered/marmousi 双双达标 <0.1 (核心要求兑现)**: marmousi 0.106→**0.046** (2.3×, late 0.16→0.067),
   layered 0.154→**0.094** (1.6×, mid 0.19→0.10). 两个非平凡家族 (有真实散射结构) 的 held-out 精度都被
   sigma=2 推到充分高. 对症第十六节诊断: 更干净的背景剥离让网络只学更小的散射残差, 中/早期大幅改善.
2. **uniform late 伪影是良性数值假象, 不代表预测力**: uniform all=0.087 全由 late=0.52 主导 (early 0.012 /
   mid 0.062 极好), 且轨迹剧烈波动 (0.34→0.081→0.13, min 0.081) 不收敛. 根因: uniform P_bg == 全场
   (bit-exact), 真实 late scat≈0, 模型在近零帧上输出 O(1) 噪声 = 分母趋零病态 (同 A+1 G2 首轮 late=2.0).
   与 sigma 无关, 是平凡家族的评分假象 —— uniform 本无散射结构可学, 其"误差"不度量算子对真实物理的能力.
3. **sigma=2 确立为 A+1 推荐配置 (精度侧)**: 部署成本权衡 = sigma=2 平滑 solve 比 sigma=4 略贵 (CFL 更严,
   保留更多细尺度), 但换来 layered/marmousi 达标. 仍是"背景 Born + 学习散射"混合范式, 非退化.

**综合: A+1 混合求解器 (sigma=2 背景) 的 layered/marmousi held-out 精度充分高 (0.094 / 0.046)**.
完整证据链: A/B/A+ oracle → G2 记忆 0.065 → G3 泛化 (sigma=4) 0.107 → G3 精度达标 (sigma=2) layered/marmousi
<0.1. uniform 的 late 伪影是平凡家族的评分假象非算子缺陷. **A+1 是可泛化且对非平凡家族精度充分高的真实算子**.
数据契约见 [[fno-acoustic-v72-experiment]].

## 第十八节: sigma=2 scaling (N=48 → N=192) —— 两族充分高, 全池不必跑 (2026-08-02)

**动机**: sigma=2 已让 layered/marmousi 达标 (N=48: 0.094/0.046)。全池 2240 sigma=2 成本高
(13.1h solve + 89GB, RAM 单进程分配 145GB 不可行)。先跑 N=192 (每族 192, 4×N=48) 验证 sigma=2 下
是否也饱和 → 若饱和则以 1/4 成本确证全池外推。4 卡分片构建 (merge_background_shards.py, HDF5_USE_FILE_LOCKING=FALSE
规避 shm 锁) + 4 卡 DDP 训练 (40ep, 每 rank 960 update)。

**sigma=2 held-out all_saved 对照**:

| N (每族) | all_saved | layered | marmousi | uniform |
|---|---|---|---|---|
| 48 | 0.076 | 0.094 | 0.046 | 0.087 |
| **192** | **0.053** | **0.082** | 0.045 | 0.031 |

time_bin (N=48 → N=192):
- layered: early 0.050→0.049, mid 0.096→0.095, **late 0.257→0.186** (数据增益主要在 late 混响)
- marmousi: 全 bin 不动 (early 0.013 / mid 0.042 / late 0.067)

**裁定 (sigma=2 下 scaling 部分饱和, 两族充分高)**:
1. **marmousi 完全饱和**: 0.046→0.045, 全 time_bin 逐位不动. 4× 数据零增益 = marmousi 散射残差在
   sigma=2 背景下已达 rank-8 映射的泛化极限, 数据量无用.
2. **layered 有真实数据增益但边际递减**: 0.094→0.082 (13%改善), 增益集中在 late 混响 (0.257→0.186) —
   更多 layered 样本让平层多次波 coda 的低秩散射映射泛化更好. 全池 2240 (再 ~12× 数据) 外推至多再降 ~0.01,
   不值 13h + 89GB.
3. **uniform 近零 late 伪影随数据消解**: 0.087→0.031 (更多训练让近零 scat 帧的随机头噪声收敛), 印证
   第十七节"uniform late 是欠训/近零病态非泛化失败"的诊断.

**综合: A+1 (sigma=2) 的实用算子精度经 scaling 确证 —— 两个非平凡家族充分高 (layered 0.082 / marmousi
0.045), 且随数据量稳健 (marmousi 饱和, layered 边际递减).** 全池不必跑: 结论 (两族达标 + scaling 行为)
已由 N=48/192 两点确立. A+1 时间-频域 Helmholtz 方向的预训练算子研究至此闭环:
oracle 可达性 → G2 记忆 0.065 → G3 泛化 (sigma=2) 两族 <0.1 → scaling 稳健. 转入实例化微调
(利用早期快照 + 高阶 LWC PDE 把已达标的算子再校准到具体实例, 见
docs/superpowers/specs/2026-08-02-aplus1-onset-snapshot-instance-adaptation-design.md).

## 第十九节: A+1 上的实例化微调 (2 早期帧) —— 信息墙依旧, 根因更清晰 (2026-08-02)

**设定**: 预训练闭环后, 用户要求"利用播前波场快照做高效实例化微调 + 高阶 LWC PDE loss"。承旧
instance-adaptation 工作 (memory: fno-acoustic-instance-adaptation) 的信息墙裁定, 但**父模型换成 A+1**
(可泛化真实算子, held-out layered/marmousi 0.082/0.045), 重测 2 早期帧能否再校准。

**实现**: `scripts/diagnose_aplus1_instance_finetune.py` — 复用 `OnsetAdaptedV5` (13k 参高容量残差头,
非部署 33 参) + 高阶 LWC `lwc84_residual(time_order=4)` + RAD 自适应采样 + `GuardedOnsetDataset` 因果守卫。
父场 = A+1 全场 = `dense_normalized (归一化散射残差) + encode_pressure(P_bg)` (sigma=2 缓存, 与 G3 同源)。
`_predict_parent` 加 background_provider 支持 (归一化空间加回 P_bg, encode 线性故精确)。

**结果 (因果 future_truth_used=False, 仅 2 早期帧)**:

| 族 | A+1 父基线 | 微调后 best | beats_parent |
|---|---|---|---|
| marmousi | 0.04512 | 0.04513 | False |
| layered | 0.08222 | 0.08249 | False |

**决定性诊断 (根因比旧工作更清晰)**:
1. **信息墙在 A+1 上依旧**: 2 早期帧微调不超 A+1 冻结父场 (两族 beats_parent=False), 复现旧 coarse-MIONet
   下的信息墙裁定 —— 但此前归因含"父模型结构瓶颈", 现父模型已是 0.045/0.082 的好算子, 排除了结构因素.
2. **根因 = onset 帧能量近零, 信息量本质不足**: 诊断 marmousi held-out 的 onset 帧 (11,12) 能量占峰值帧
   (frame 99) 仅 **0.02%** (帧 30 才 48% / 帧 50 才 95%); onset 2 帧携带全场 **3e-5** 的能量.
   A+1 父场在这 2 帧的 relL2=3.87 (相对误差大但绝对能量微不足道, 因优化的是全场 relL2). obs_loss 卡在
   2-3.4 不降 = 高容量头都无法从近零帧提取有用约束. **不是父模型不好, 是 2 近零早期帧信息量本质不足**,
   与 CEILING 研究 (K~48 才够) 完全一致.
3. **高阶 LWC PDE 项已就位但无从发力**: time_order=4 的物理残差监督未观测时刻, 但 2 帧锚点信息太弱,
   PDE 外推缺乏起点约束 (与旧工作"2 帧契约下采样/物理项差异不改结果"同理).

**综合**: A+1 父模型 + 高阶 LWC + RAD + 高容量头, 在严格 2 早期帧契约下仍不超父场 = 信息墙确证为
**观测契约信息量问题, 非父模型能力问题**. 安全契约成立 (未恶化父场, 因果审计通过). **出路 (CEILING 指引)**:
放宽观测契约到 K~48 含中期高能帧 (在线同化场景), 已有基础设施 (onset_indices 可扩展) 直接可测.
高阶 LWC PDE loss 升级 (第七节工作) 为放宽契约后的物理外推就位.

## 第二十节: 纯PDE自监督微调不可行诊断 —— PDE残差选不出真值 (2026-08-02)

**设定**: 用户澄清部署可获早期真值帧(数值solver求frame0-80, ~20%时间步, CPML与训练一致), 微调监督
用纯全时空高阶LWC PDE残差(time_order=4)把解外推到晚期。目标: PDE残差微调A+1提高精度。

**go/no-go探针链 (marmousi held-out)**:
1. **naive优化探针** (固定真值frame0-80, 自由优化81-400, 只最小化自归一PDE残差): late_relL2爆到1e6,
   PDE loss仅0.95→0.94几乎不降 = 病态发散.
2. **CFL诊断**: CFL=vmax·dt_saved/dx=**0.58<1** (saved-dt其实稳定, 推翻"超CFL必炸"); 真值场晚期
   modified-eq残差rel=**0.0025** (真值确实满足saved-dt离散PDE, 非欠定).
3. **裸LWC递推** (真值frame79,80前推81-400): relL2=0.84 = 稳定但不准, 根因裸递推缺CPML吸收边界
   (真solver有CPML, 波在存储域边界反射累积).
4. **决定性诊断 (A+1场vs真值场的PDE残差)**: A+1场全场relL2=0.045 (离真值仅4.5%), 但
   **A+1场PDE残差=0.0068 ≈ 真值场PDE残差=0.0066** — 几乎相同. 晚期(200-400)A+1 relL2=0.061.

**裁定 (纯PDE自监督不可行)**: **PDE残差无法区分A+1场与真值场** — A+1的4.5%误差在PDE残差上不可见,
因A+1已是近似满足波动方程的光滑场, 已落在PDE残差极小值附近 (梯度≈0). 纯PDE残差微调改善不了A+1.
物理根源: **PDE残差是必要非充分** — 波动方程有无穷多解 (不同初值/边界都满足PDE), PDE选不出"这个
具体记录"的真值. 唯一钉住具体记录的是**早期真值帧作数据约束** (初值/边界条件). 纯物理自监督缺这个锚,
PDE正则对已光滑的A+1场无区分力.

**修正的方法方向**: 微调监督必须 = **早期真值帧(frame0-80)数据match (强锚定这个记录) + PDE残差正则
(保物理一致)**, 而非纯PDE. 早期帧不再是近零onset(2帧信息3e-5的信息墙), 而是能量充分的frame0-80
(峰值97%). 关键开放问题: A+1晚期已0.061, 早期帧数据match能否通过"钉住早期真实波前"传播改善晚期?
需实测 (早期帧match是新信息: A+1早期帧本身有误差, 真值早期帧提供A+1没有的实例特定信息).

## 第二十一节: 早期窗真值帧 + PDE 正则微调 —— A+1 早期已准, 无改善空间 (2026-08-02)

**方法 (承第二十节修正)**: `scripts/diagnose_aplus1_early_window_finetune.py` — A+1 父场 + 早期真值帧
(frame 0..80, stride 4 = 21 帧, 数值 solver 只求早期时间步的部署合法信号) 数据 match + 高阶 LWC PDE
(time_order=4) 正则. EarlyWindowAudit 强制只读 <=k_end 帧 (因果). 13k 高容量残差头. 判据: held-out
(frame 81-400) 是否 < A+1 冻结父场.

**结果 (2 族)**:

| 族 | A+1 父 held-out | 微调后 best | **A+1 父 早期帧(0-80)** | beats_parent |
|---|---|---|---|---|
| marmousi | 0.0491 | 0.0504 | **0.0039** | False |
| layered | 0.0970 | 0.1041 | **0.0336** | False |

**决定性诊断 (推翻"早期帧是新信息"的假设)**: **A+1 父场在早期帧 0-80 已经极准** (marmousi 0.4% /
layered 3.4%). 第二十节说的 A+1 4.5% 误差是**全场**平均, 但按时间分解: 误差几乎全在晚期, 早期帧 A+1
已近完美. 所以早期真值帧**几乎不含 A+1 没有的信息** — 用它做 match 只是把已经对的 residual head 往对的
地方推, 反而轻微扰动晚期 (0.049→0.050 / 0.097→0.104).

**综合裁定 (实例化微调在 A+1 上无改善空间)**: A+1 误差的时间结构 = 早期准 (0.4-3.4%) / 晚期偏
(0.06-0.10), 是"从正确早期演化到晚期"时 rank-8 低秩散射表示的固有累积偏差. **部署可得的信息 (速度/源/
PDE/早期帧) 都无法钉住这个晚期偏差**: 早期帧 A+1 已经对 (无新信息), PDE 残差选不出真值 (第二十节, 必要
非充分), 中晚期真值不可得 (部署约束). **信息墙的最终形态**: 不是缺容量、不是缺早期观测, 而是"晚期偏差
所需的信息 (中晚期结构) 在部署时物理不可得". 安全契约始终成立 (residual_gate 可回滚, 因果审计
future_truth_used=False). **A+1 冻结父场即最优部署形态** (layered 0.082 / marmousi 0.045 全场,
早期 0.4-3.4%), 实例化微调在此算子质量下不再有正收益.

**要改善晚期只能回到预训练侧** (已闭环的方向): 更高 rank / sigma=2 背景 / 更多数据 — 但第十六/十八节
已证 marmousi 数据饱和、layered 边际递减. A+1 的晚期精度是当前架构的收敛点.

## 第二十二节: 晚期高秩专用头继续预训练 —— 秩容量非瓶颈, backbone特征才是 (2026-08-02)

**动机+设计**: 诊断 A+1 晚期残差空间秩 32-51 (early7/mid19/late32) 而 A+1 用 rank-8, 判定晚期欠秩。加
零初始化低频高秩晚期头 (late_rank=32, 最低24 bin, `_HelmholtzSynthesisField` 加 late_basis_head +
零初始 late_cos/sin_mix + 零初始 late_mix_condition), warm-start A+1 N=192 sigma2 best (strict=False,
仅 late-head 留新), 4卡 DDP 继续预训练 (低 lr: backbone 2.5e-5/dense 5e-5/local 2.5e-4)。零初始逐位契约
验证通过 (late_rank=32 vs rank-8 输出 max|diff|=0, warm-start baseline=A+1 精确)。

**结果 (N=192 sigma2, 40ep, held-out all_saved, 轨迹 u144→u768 完全平台)**:

| | A+1 基线 | + late-head | 变化 |
|---|---|---|---|
| layered late | 0.186 | **0.161** | -13% (小幅) |
| layered mid | 0.095 | 0.095 | 0 |
| layered all | 0.082 | 0.078 | -5% |
| marmousi late | 0.067 | **0.068** | 0 (std 5e-5 全平台) |
| marmousi all | 0.045 | 0.044 | 0 |
| agg | 0.045 | ~0.043 | 微降 |

**决定性诊断 (复测训练后晚期残差秩)**: layered late 残差秩@90% 32→**29**, marmousi 32→**31** —— 加了
32 秩容量但**残差秩几乎没被吸收**。这证明晚期偏差**不是 late-head 表达容量不足, 而是 backbone 的 render
特征本身不含晚期多次波所需的高秩空间信息**。late-head 从 render 特征生成基场 (`late_basis_head` 是 render
的 1x1 conv), render 特征里没有的结构, 加多少秩也提取不出来。

**裁定 (秩容量非瓶颈)**: "目标高秩 ≠ 可从现有特征生成" —— 与 [[fno-acoustic-v72-experiment]] 容量梯
"加参数无用" 同源, 但这里更精确: 不是解码器容量, 是 render backbone 的**特征信息**不足以支撑晚期高秩合成。
零初始契约成立 (early/mid 不退化, 安全), late-head 对 layered 有小幅正效果 (0.186→0.161, backbone 特征
恰好含少量 layered 平层结构), 对 marmousi 无效 (复杂散射的晚期多次波结构不在 render 特征里)。

**根因链完整**: A+1 晚期偏差 → 不是秩不足 (加秩残差秩不降) → 是 render backbone 特征不编码晚期多次波结构。
**要真正改善晚期须让 backbone 编码晚期结构** (而非在合成头加秩): 候选 = (a) 给 U-Net render 加晚期时间
条件输入 (让 render 随查询时刻变, 但破查询不变/加成本); (b) 更深/更宽 backbone 提升特征信息 (第九节容量
梯已示宽度饱和, 存疑); (c) 接受 A+1 当前晚期精度为架构收敛点。**结合第十六/十八节 (数据饱和) + 二十/
二十一节 (部署侧无空间) + 本节 (加秩无效): A+1 的晚期精度 (layered late 0.16-0.19 / marmousi 0.067) 是
当前 render-then-synthesize 架构族的收敛极限, 突破需 render backbone 本身的信息改动, 非合成头或微调。**

## 第二十三节: 多到达WKB假设证伪 + 晚期根因精确定位 (2026-08-02, 零训练oracle)

**假设(第二十二节后提出)**: 上轮加秩失败可能因A+1 WKB用单走时τ的e^{iω(t-τ)}载波只表达单到达, 晚期多次波
(直达+反射, 多到达时间)被单τ坍缩, 秩再高补不回丢的多到达相位。

**探针**: `docs/helmholtz/probe_multi_arrival_wkb.py`, 零训练oracle(容量无限每像素自由系数, 测ansatz表达力
非可学性)。对真值晚期场(frame201-400)用ridge正则batched最小二乘拟合K到达WKB合成
Σ_{k<K}Σ_j[a_kj cos(ω_j(t-τ-Δ_k))+b_kj sin], τ=eikonal首到达固定, Δ_k延迟网格搜索。K=1(A+1单到达)
vs K=2,3(多到达)。

**结果(marmousi/layered held-out, nf=48)**:

| K | marmousi late relL2 | layered late relL2 | amp_rank95 |
|---|---|---|---|
| 1 (单到达=A+1式) | **0.0032** | **0.0012** | 38-39 |
| 2 (加1反射) | 0.0031 (gain 9e-5) | 0.0012 (gain 1e-5) | 63-69 |
| 3 | 0.006 (变差/过拟合) | 0.010 (变差) | 100+ |

**裁定(多到达假设证伪)**: **单到达WKB表征已充分**(oracle晚期 0.1-0.3%), 多到达K=2几乎零增益、K=3反而
过拟合变差。我第二十二节后提出的"WKB单走时是晚期瓶颈"假设**错误**——单τ载波足以表达晚期场到 0.1-0.3%。

**晚期根因精确定位(oracle副产物)**: 单到达oracle的振幅场 amp_rank95=**38-39** (A+1用rank-8)。所以晚期
瓶颈=**需要rank~38的振幅场而A+1 rank-8截断**, 不是WKB载波、不是多到达。结合第二十二节(加rank-32 late-head
后残差秩没降): 完整根因链 = 晚期需rank~38振幅场 → (a)A+1结构rank-8不足 (b)加late-rank-32头也没用因
**backbone render特征本身生成不出这些高秩平滑振幅场**。**瓶颈是render backbone的高秩特征生成能力**, 不是
合成头秩数(加了不吸收)也不是载波形式(单到达够)。

**综合**: 晚期改善的唯一有效杠杆 = 提升render backbone本身的特征表达(生成rank~38振幅场的能力), 而非
合成头/载波/微调/数据。这比第二十二节"backbone特征不含晚期信息"更精确: backbone需生成的是**高秩(~38)平滑
振幅场**, 当前U-Net render(4级pyramid, width128)在rank-8截断下达不到。候选=更宽render/多分辨率谱混合
(FNO式全局)/render输出直接高秩基。但须权衡: oracle rank38但A+1收敛rank8=可学性(映射)可能仍卡在rank8
(第九节"目标简单≠映射可学"), 需先探"backbone能否学出rank38振幅"再改架构。

## 第二十四节: 评估目标修正 + 晚期根因最终锁定=render特征无多次反射几何 (2026-08-02, 决定性)

**关键修正(推翻§22/§23的目标)**: §22/§23测的"晚期残差/秩"用的是 truth 或 truth-A+1全场, 但 A+1 =
物理背景 P_bg + synthesis头学的散射残差 scat=truth-P_bg。**核验: A+1晚期精度0.061几乎100%来自P_bg**
(P_bg单独晚期=0.061=A+1全场0.061), 晚期 scat 只占 truth 6.1%能量。所以 synthesis头(render特征驱动)的
真实目标是 scat 不是 truth。§22加秩/§23多到达/§21微调都在错误目标(truth,含P_bg物理结构)上评估。

**问对目标的go/no-go探针(`docs/helmholtz/probe_render_feature_rank.py --background-cache`)**: 从冻结A+1
render特征[128,H,W]做无限秩线性映射拟合晚期 **scat**:

| 目标 | marmousi | layered |
|---|---|---|
| feature-linear 拟合 scat 晚期 relL2 | **0.977** | **0.975** |
| A+1 synthesis头(rank8)对scat晚期 relL2 | **1.001** | — |

**决定性裁定(render特征墙, 问对目标后成立)**: render特征无限秩线性都几乎无法生成晚期scat(relL2 0.98≈
零解释力), A+1 synthesis头对晚期scat贡献=0(relL2 1.0=比零预测还差)。**A+1晚期0.061纯靠P_bg背景,
神经网络synthesis头对晚期毫无贡献**。前几轮加秩/late-head/多载波必然失败——不是它们的问题, 是 render
特征从源头不含晚期散射结构。

**晚期根因最终锁定(物理层面)**: 晚期 scat = 波在层间**多次反射**(multiples/coda)产生; 而 render 输入的
相位锚只有**首到达走时 τ**(单次直达几何, eikonal first arrival)。网络看不到多次反射路径的几何, 自然
生成不出多次波散射场。**瓶颈 = render 输入缺多次反射几何先验**, 不是synthesis头容量/秩/载波/微调/数据。

**唯一可行的架构方向(明确)**: 给 render 输入**多次反射的几何先验**, 让特征编码晚期散射结构。候选:
(a) 相位锚从单τ扩展到**多次反射走时场**(首到达+一次反射+二次反射的eikonal/ray走时, 项目已有RayTravelTime
可扩展); (b) render用**全局谱层(FNO)**捕捉全域多次反射相干(局部U-Net感受野虽近全局但特征未编码反射几何);
(c) render输入加**速度界面/反射系数图**(多次波源自波阻抗界面)。**这是首个有物理机制依据的晚期架构方向**
(区别于§22-23被证伪的秩/载波), 但属render backbone重构=新一轮较大改动。**当前A+1(晚期靠P_bg达
layered0.082/marmousi0.045)是"物理背景+散射残差"架构在synthesis头层面的收敛终点; 晚期神经贡献=0是因为
render看不到反射几何。**

## 第二十五节: scat上重测多到达=多次反射走时方向也证伪, 唯一瓶颈=render高秩生成 (2026-08-02)

**动机**: 第24节指"给render加多次反射几何"是唯一方向, 但第23节多到达oracle用错目标(truth含P_bg)。
在正确目标 scat=truth-P_bg 上重测多到达(probe_multi_arrival_wkb.py --background-cache), 决定是否值得
实现多次反射走时。

**结果(scat晚期, nf=48)**:

| K | marmousi scat late | layered scat late | amp_rank95 |
|---|---|---|---|
| 1 (单到达) | **0.0070** | **0.0025** | 37 |
| 2 (加反射延迟) | 0.0068 (gain 2e-4) | 0.0025 (gain 6e-5) | 66-69 |
| 3 | 0.021 (变差) | 0.015 (变差) | 89-90 |

**裁定(多次反射走时方向也证伪)**: 即使在正确的 scat 目标上, 单到达WKB自由系数oracle已充分
(scat晚期0.25-0.7%), 多到达零增益K=3过拟合。**晚期scat不需要多到达相位**——单τ载波+rank~37振幅场足以
表征。所以第24节候选(a)多次反射走时锚**无效**(走时/相位不是问题)。

**晚期瓶颈最终唯一锁定(所有表征假设逐一证伪)**:
1. 载波/走时/多次反射几何: 单到达充分, 多到达无用 (§23 truth + §25 scat 两次证伪) → 不是相位问题
2. 合成头秩数: 加rank/late-head残差秩不降 (§22) → 不是头容量
3. render特征: 无限秩线性拟合scat=0.98 (§24) → **render特征不含rank~37的scat振幅结构**

**= 唯一瓶颈 = render backbone生成高秩(~37)散射振幅场的能力**, 与走时/载波/相位/几何先验无关, 纯粹是
render特征的表达力/有效秩。当前U-Net render从静态记录条件生成的128维特征, 有效秩不足以线性张成rank-37
的scat振幅场。**剩余唯一方向 = 直接提升render特征生成能力**(更宽/更深render, 或FNO谱层增有效秩), 而非
任何几何/相位/载波/秩-头改动。**下一步go/no-go = 测render特征的有效秩**: 若render特征有效秩本身<37, 则
证明须扩render容量; 这决定"扩render宽度能否解决"。这是承接的最基础可学性问题。

## 第二十六节: 晚期根因彻底锁定=render特征有效秩塌到8 (2026-08-02, 收敛点)

**探针**: SVD render特征[128,HW]的有效秩。

**结果(两族一致)**: render特征有效秩 **@90%=5 / @95%=8 / @99%=26** (总128维但塌缩), 晚期scat需rank~37。

**根因彻底锁定(不可再归因)**:
1. render特征有效秩仅~8(@95%), 远低于晚期scat所需rank~37 → 特征张不成scat振幅场(feature-linear 0.98)
2. **render有效秩8 == A+1合成头rank-8**: 非巧合——render backbone从静态记录条件(速度+源+首到达走时τ)
   生成的特征本身只有rank-8有效信息, 合成头rank-8正好用尽; 加更多秩(§22 late-rank32)无更多特征可用,
   多载波(§23/25)无用(不是相位), 微调(§21)无用 → **所有失败的共同根=render特征秩塌到8**。
3. 物理原因: render输入是低秩静态量(1速度场+1源+1走时场), U-Net局部卷积无法从低秩静态输入生成高秩
   多次散射结构; 晚期scat的rank~37需要全局耦合/更丰富输入才能诱导。

**晚期改善的最终架构裁定(唯一路径)**: 提升render特征有效秩 8→37+。两条候选:
(a) **FNO全局谱层** 替换/增强局部U-Net render (全局耦合可从低秩输入生成高秩特征, 局部卷积不能);
(b) **更丰富render输入** (诱导高秩的额外条件通道)。
走时/载波/相位/合成头秩/微调/数据 全部已证伪(§16-25), render特征有效秩是唯一且最终的瓶颈。

**全链条闭环总结(A+1时间-频域Helmholtz + 晚期攻坚)**: 
- 预训练: oracle<5%可达 → G2记忆0.065 → G3泛化sigma2 layered0.082/marmousi0.045(两族<0.1, scaling稳健)
- 晚期偏差(layered late0.16/marmousi0.067)攻坚: 部署微调(信息墙§19-21) + 训练侧加秩(§22) + 多载波
  (§23/25) + 早期帧(§24) 全证伪; 根因收敛到 render特征有效秩塌到8 (§26)。
- A+1(晚期靠物理背景P_bg)是"物理背景+散射残差, 合成头rank-8"架构族的收敛终点; 进一步晚期改善须重构
  render backbone提有效秩(FNO谱层/更丰富输入)=新一轮研究规模。

## 第二十七节: 秩塌缩定位在U-Net处理(输入秩66→输出8), 非信息源 (2026-08-02)

**文献(本地arxiv API 2024-2026)**: render特征秩塌8 = **spectral bias**(网络偏好低频/低秩), 多篇独立
工作确证且针对波散射/Helmholtz: MscaleFNO(2024/2026, 并行多FNO输入按尺度缩放降spectral bias, 高振荡
介质→散射场), SirenFNO(2026, SIREN正弦激活mode-wise核克服频率截断), SVD-NO(2025, 显式SVD低秩核=限秩
不适用)。

**关键go/no-go(区分文献方向a架构vs b更丰富输入)**: 测U-Net render的输入vs输出有效秩:

| | 输入有效秩 @90/95/99 | 输出有效秩 @90/95/99 |
|---|---|---|
| marmousi | 42 / **66** / 107 | 5 / **8** / 26 |
| layered | 43 / 67 / 107 | 5 / 8 / 26 |

**决定性裁定(秩塌缩在U-Net处理, 非信息源)**: unet输入有效秩@95%=**66 >> 晚期scat所需37**——高秩信息
**已在输入里**(相位特征/dense_propagation_features丰富); 但U-Net把66压塌到**8**。**排除文献方向(b)更丰富
输入**(输入已够), **锁定(a)架构——U-Net处理导致秩塌缩**。比笼统"改backbone"精确: 不需换整个render, 只需
**打破U-Net的秩塌缩瓶颈**。

**塌缩机制(推断)**: U-Net pyramid下采样(avg_pool到13×13最深)+有限通道在深层压缩秩; 局部卷积的spectral
bias偏好低秩输出。MscaleFNO多尺度并行正是保高频/高秩不塌缩的已验证机制。

**精确下一步方向**: 打破U-Net秩塌缩, 让输出保留输入已有的高秩(66→保到37+)。候选(按成本/对症):
(a) render输出前加**多尺度/谱层旁路**(MscaleFNO式, 保高频分量, 可插拔零初始warmstart A+1);
(b) 减轻pyramid下采样深度或加高分辨率skip保高秩; (c) render输出通道/合成头rank从8提到40配合。
判据: render输出有效秩@95%从8升到>=37 + 晚期scat feature-linear从0.98降。**首个精确到"打破U-Net秩塌缩"
且信息源已证充分的可执行方向**(区别于§16-26逐一证伪的载波/秩/走时/微调)。

## 第二十八节: 多尺度谱旁路结果 + 研究收尾 (2026-08-02)

**多尺度谱旁路(承第27节, MscaleFNO式打破U-Net秩塌缩)**: 实现`_MultiScaleSpectralBypass`(并行3个
modes 8/16/32谱卷积AxisFactorizedComplexSpectralConv2d, 从render高秩输入66提多尺度分量加到U-Net输出,
零初始gate warmstart no-op)。**首版冷启动死锁bug**: gate+mix双零初始→梯度互锁, 谱分支永不学(u48-192
平台化layered late 0.16=退回纯A+1, gate=0/mix范数=0确认)。**修复**: mix保非零初始只gate零初始(no-op契约
保持: gate=0→输出0; gate梯度=mix_out·(...)≠0可学)。**修复版结果(N=192, 96 update)**: gate爬到仅
0.00026(极慢), render输出有效秩@95% 8→**9**(几乎没动, 目标37), layered late 0.176→0.168(未破0.16平台),
marmousi late 0.068不动。**裁定: 零初始gate+继续预训练范式下谱旁路提秩太慢不实用**——gate从0爬升极慢,
96步有效秩仅8→9, 外推960步也难达37。非方向严格证伪(gate未充分训练), 但在warmstart继续预训练范式下
不实用; 真正验证需更激进方案(gate正初值破no-op契约/从头训练非warmstart)=另一研究周期。

**研究收尾裁定**: 
- **核心成果完整可交付**: A+1混合求解器(查询不变时间-频域Helmholtz合成 + 物理平滑背景) G3 held-out
  layered 0.082/marmousi 0.045(两非平凡族<0.1), 泛化gap 3×→1.7×, scaling稳健。全链证据: oracle<5%可达
  →G2记忆0.065→G3泛化两族<0.1。
- **晚期极限根因精确诊断(有价值的负结果)**: 晚期偏差=render特征有效秩塌到8(输入66→输出8, 需37),
  经7方向系统证伪(部署微调§19-21/加秩§22/多载波§23,25/早期帧§24/纯PDE§20)锁定; 谱旁路方向有文献支撑
  但warmstart范式下提秩太慢(§28)。render-then-synthesize架构族在合成头层面的收敛终点。
- **收尾理由**: 两非平凡族已达标, 晚期(layered late 0.16)是边际问题对全场agg贡献小; 晚期已投入充分,
  继续调gate参数边际收益<投入。**交付A+1 + 完整诊断链 + TGRS论文(paper/tgrs_helmholtz_operator/)**。
- **若未来重启晚期**: 唯一有据方向仍是提render有效秩, 但需跳出warmstart继续预训练(从头训练带谱旁路,
  或gate正初值), 属新研究周期。

## 第二十九节: 实例化微调优化调研 + 早期↔晚期信息解耦 (2026-08-02, 决定性)

**文献调研(本地arxiv API)**: 实例化适配的最新范式绕开权重微调——In-Context Operator Networks(ICON,
数据prompt条件化, 不更新权重)/ Neural Operator Processes(NOP, 部分观测概率条件化)。它们改变"如何用
观测", 但不改变"观测含多少信息"。

**关键go/no-go(决定ICON/NOP方向是否有据)**: 可观测的早期真值帧能否预测晚期scat? 用shared时间线性
算子 W[Tl,Ke] (早期时间序列→晚期, 跨像素共享, 参数少数据多不过拟合) 拟合:

| 族 | 早期(0-80)→晚期scat 线性relL2 | 早期-晚期能量相关 |
|---|---|---|
| marmousi | **1.00** (=零预测) | -0.01 |
| layered | **1.00** (=零预测) | 0.10 |

**决定性裁定(早期↔晚期信息解耦)**: 早期真值帧无法线性预测晚期scat到任何优于零预测的程度; 能量几乎
不相关。**晚期散射信息根本不在可观测的早期帧里**。**物理解释(因果)**: 晚期多次波来自波传到深层界面
反射返回, 早期(波前刚出发)波场尚未"访问"那些界面, 故早期观测不含晚期反射信息——晚期依赖波未来将
经过的介质区域, 早期因果上观测不到。

**对所有观测条件化方法的裁定**: ICON/NOP/任何观测条件化都救不了晚期——它们改变观测的使用方式, 但晚期
信息不在观测里, 再巧的条件化也变不出不存在的信息。**从信息论封死部署侧(微调+条件化)的晚期优化**:
可得的早期帧无晚期信息(本节), 含晚期信息的中晚期帧不可得(部署约束, §24), 两者夹击=部署侧实例化适配
对晚期无解。

**综合(实例化微调最终结论)**: 之前§19-21的信息墙不是"微调方式差"或"没用in-context", 是早期观测与晚期
散射在信息上因果解耦。实例化微调的正当价值仅剩安全契约(不恶化父场, §17-18 instance-adaptation memory)
和"若中晚期稀疏帧可得则K~48可达0.06-0.14"(CEILING研究)——但后者需要部署不可得的观测。**部署侧晚期
优化封闭; 晚期改善只能回预训练侧(render有效秩, §26-28), 且需跳出warmstart(新周期)。**

## 第三十节: 廉价晚期快照来源(粗网格solve)证伪 (2026-08-03)

**需求(用户澄清)**: 实例化微调需(1)总代价<传统求解器1/5 (2)若用晚期快照须不重跑完整求解器廉价获取。
文献确证范式=粗求解器+神经校正(Nguyen&Tsai 2023 / end-to-end 2024)+multi-fidelity FNO。候选廉价晚期
快照来源: 粗网格solve / P_bg平滑solve / 更小sigma。

**实证(marmousi held-out, 全分辨率solve基线20.7s对真值relL2=0.006)**: 2×粗网格(201², dx10m, dt×2)
solve: **10.4s(仅加速2.0×, 未达5×)**, 晚期relL2=**0.377(比A+1的0.061差6×)**。

**裁定(粗网格晚期快照证伪, 两问题皆致命)**: ①加速不够——2×粗网格仅2×(非理论8×, GPU小网格并行利用低
+CPML开销主导), 要5×需更粗精度更差; ②精度远差——dx10m频散致晚期0.38, 比A+1已有0.061差一量级, 用0.38
粗快照"改善"0.061方向错(快照本身比A+1差, 微调只会拉低)。

**根本两难(穷尽廉价晚期源)**: 晚期精度需细网格(=全求解器成本), 所有廉价近似都有频散(比A+1差): 粗网格
✗(慢+频散); P_bg平滑solve晚期=0.061=A+1(无新信息, §14); A+1自身✗(循环); sigma→0平滑=全速度全求解器
(不省)。**这正解释A+1为何用平滑背景而非粗网格**——平滑背景在细网格跑(无频散)只丢细尺度散射, 粗网格省
成本但引入频散比丢细尺度更糟。**结论: "不重解方程得到优于A+1的晚期快照"在本设定无廉价解**; 晚期信息
要么需全求解器(不省), 要么廉价近似有频散(比A+1差)。实例化微调用晚期快照改善晚期=无廉价路径。

**对用户两约束的诚实回答**: 
- 加速≥5×: A+1神经forward本身是ms级 vs 全求解器20.7s = 已 >>5×加速(正演本身达标); 实例化微调若不需
  额外solve(纯神经微调)也保持加速。
- 晚期快照廉价来源: 无——粗网格频散使其比A+1差, 平滑背景无晚期新信息。**晚期改善不能靠廉价晚期快照,
  只能回预训练侧(render有效秩, §26-28)。**

## 第三十一节: 解析均匀背景替代P_bg (用户想法, 2026-08-03)

**用户想法**: P_bg换成与真速度相关的均匀模型解析解(时域Green函数), 免数值求解器→P_bg成本21s降~0.1s。
背景动机(成本账实测): A+1部署=P_bg solve(21s)+推理(6.5s)=27.5s>全求解器20.7s, **算上P_bg后A+1比传统
求解器还慢**, P_bg是A+1速度瓶颈(全分辨率平滑solve, 粗网格因波场频散不可加速)。

**实现**: 2D时域声波自由空间Green G(r,t)=H(ct-r)/(2πc²√(t²-(r/c)²))与存储源子波时域卷积(FFT), 每记录
等效均匀速度c(源局部/均值/中位)。**用时域Green(非§10频域Hankel)对症"时域声波方程"**(§11已证数据非
时谐Helmholtz)。成本**0.1s**(vs P_bg 21s, 快200×)。

**结果(源局部c, 相对真值场relL2)**:
| 族 | c_used | all | early | mid | late |
|---|---|---|---|---|---|
| marmousi | 1500 | 0.92 | 0.64 | 0.86 | 1.04 |
| layered | 3392 | 0.89 | 0.68 | 0.94 | 1.06 |
| **uniform(真均匀!)** | 3641 | **0.77** | 0.67 | 0.80 | 0.86 |

**裁定(方向early有价值但整体不足, 且解析-离散失配未解)**: ①**探针正确性存疑**——uniform族(本身均匀
速度)解析均匀Green应near-perfect(§10数值uniform-c0 self-check bit-exact 0.0), 但实测仅0.77→时域Green
实现有解析-离散失配(数据源=分布式source_map+8阶FD+binomial5低通+decimate2 vs 理想点源连续Green), 同§10
放弃解析改数值的根因; ②即便如此, 非均匀族early仅0.64-0.68(mid/late更差>0.9)显示**均匀背景对非均匀介质
本就弱**——缺(a)速度变化的波前变形(均匀波前是圆, 真实随v(x)变形)(b)界面反射(均匀无界面无多次波, late
>1.0=比零还差同§10 late退化); ③比朴素算子0.639还差(0.77-0.92), 远不如平滑P_bg(0.045-0.082)。

**综合**: 用户想法成本方向正确(解析免求解器200×加速), 但**均匀背景精度不足**: 解析-离散失配(uniform都
到不了0)+均匀模型物理缺陷(无波前变形/无反射)。要补波前变形需eikonal走时τ(x)=退化成A+1的WKB synthesis
(已泛化不足需P_bg补, 循环); 要补反射需真实介质solve(=P_bg, 不省)。**均匀解析背景替代P_bg不成立**——
它省成本但丢掉P_bg保留的速度变化传播+界面反射结构。**A+1的P_bg速度瓶颈无廉价解析替代**。

## 第三十二节: 谱注入提升有效秩(8→14)但晚期未破——瓶颈转移到合成头rank (2026-08-03, 关键诊断)

**谱注入run(gate=0.1全开, 5谱分支8-64, N192 sigma2继续预训练, 中途u240)**:
- early/mid纹丝不动=A+1(谱注入不扰乱已准早中期); layered late 0.286(噪声)→0.189→**0.161平台**(u192-240);
  layered全场0.078(略优A+1 0.082); marmousi late 0.068不动; gate 0.1→0.056(优化器在关小)。
- **复测render有效秩: @95% 8→14, @99% 26→36(几乎达目标37)**——谱注入**确实提了有效秩**(接近翻倍),
  不像上一轮零初始gate(8→9没提)。

**决定性诊断(推翻"有效秩是唯一瓶颈")**: **render有效秩提到14(近目标), layered late却仍平台0.16**。
说明有效秩提升是**必要非充分**——render特征秩够了, 但下游没转化成晚期精度。**瓶颈转移**: 
- **合成头是rank-8**(basis_head出R8基场), render特征秩升到14, 但**rank-8合成头又把它截断回8**。
  高秩特征进来, 低秩合成头截断→晚期高秩振幅(需37)仍出不来。
- 这**重新解释§22加late-rank头失败**: 那时render特征还是rank-8(没提), 加合成头rank无源可用; 现在
  render特征rank-14了, 配合提合成头rank才可能有效——**两者必须同时提**(render谱注入提特征秩 + 合成头
  提rank用这些特征)。

**针对性改进方向(用此诊断)**: 谱注入(已验证提render秩8→14)+ **同时提合成头rank R8→更高**(如R16-24, 
配合render新增的高秩特征)。§9曾测R16在纯低秩(render仅rank8)下回欠定, 但那是render特征不足时; 现在render
谱注入供了rank-14特征, 高rank合成头有源可用, 可能不再欠定。这是"render提秩+合成头提rank"的联合改进,
比单提一头(前几轮都单提, 各自被另一头卡)更对症。marmousi late 0.067不动仍是另一层问题(其晚期结构可能
根本低能量, 见§17 uniform late伪影类比)。

## 第三十三节: 联合提秩与逐分支渐进谱旁路仍停在同一晚期平台 (2026-08-03)

**联合提秩对照**: `helmholtz_g3_specinj_laterank32_N192_ddp4` 同时启用谱旁路与
rank-32 late synthesis head，并在完全相同的 N=192、sigma2、三族 held-out 协议下训练 960 updates。
其最佳 fixed aggregate 为 **0.04098**，但 layered late 始终约 **0.1607**、marmousi late 约
**0.0675**；最终 all-saved aggregate 为 **0.04195**。因此第三十二节提出的“render 提秩 + synthesis
提秩必须联合”已被同协议实验直接证伪：低秩 synthesis head 不是平台的充分解释。

**逐分支渐进谱旁路**: `helmholtz_g3_specinj_perbranch_N192_ddp4` 为 modes 8/16/32/48/64 分支分别设置
频率递减正 gate，检验单 gate 被优化器整体关小是否掩盖高频分支。训练健康运行至 update 768 后按负结果
提前停止。最后六次独立 fixed-panel 评估（updates 528--768）中：layered all 仅在
**0.07782--0.07789**，layered late 仅在 **0.16061--0.16100**，marmousi all 在
**0.04423--0.04425**，marmousi late 在 **0.06754--0.06756**。连续 240 updates 无趋势，且与
laterank32/A+1 平台逐位一致，故继续到 960 不具信息增益。该 run 是有意终止的负实验，不存在
`terminal.json`；最后可复现快照为 `checkpoints/update_0768.pt`，裁定写入 `aborted_verdict.json`。

**最终裁定**: 在 A+1 warm-start/继续预训练范式内，单 gate 谱旁路、谱旁路 + rank-32 late head、以及
逐分支渐进 gate 均不能把非平凡族晚期误差拉离同一平台。render effective rank 的提升是可测现象，但不等价
于可用的晚期散射信息，也不能据此把因果瓶颈简单转移给 synthesis rank。至此 warm-start 提秩路径关闭。
若仍要对晚期作最后一次架构检验，唯一尚未混入该约束的对照是：保留相同已训练前端，但让 fresh Helmholtz
render、谱旁路与 synthesis head 从初始化起共同训练；必须独立 artifact、相同 held-out gate，不能从 A+1
整模型继续预训练。若该对照仍停在 0.16，则应结束晚期攻坚并保留 A+1 作为已验证交付模型。
