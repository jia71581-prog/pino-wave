## finding

**F1 — 目标函数与门指标不是同一个统计量,且门指标被"近零能量帧"支配。** 训练损失 `exact_microblocked_loss` 优化的是**逐帧相对 L2 的平方均值**(权重 1/‖truth_t‖²),门 `_family_gain` 用的是**未平方的逐帧相对 L2 均值**。两者在重尾分布下方向不一致(Jensen),优化器会牺牲大量中间帧去压一个巨大帧。uniform 记录 parent 的 late-third 逐帧相对 L2 = 49.2(候选 63.5),而全局能量相对 L2 只有 0.076/0.082 —— 门指标已由波前离场后的 PML 残噪帧主导。

**F2 — 族间符号不一致不是族属性,是能量区段属性,且在 oracle 里就已存在。** 同一 rank16 oracle:raw POD/逐点 LS 下 uniform 逐帧口径 = **−1.639**,但同一记录 `residual_energy_capture = 0.2317`(等价全局能量口径 ≈ +12.3% 降幅);加权臂下 uniform 又变成 **+0.2225**。同一容量在三种口径下给出 −164%/+12%/+22%。所以"rank16 对 uniform 无能力"这一结论**不成立**,是口径造成的。

**F3 — smoke 的 oracle 常数存在跨记录/跨合成样本迁移缺陷。** 0.2225/0.3733/0.2979 硬编码于 `r16_dscp_engine_v3.py:320`,来源是 `r4e7_raw_weighted_pod_fresh3_confirmation_v1` 的 train_uniform_00102 / train_layered_01032 / **synthetic**_train_marmousi_fresh_v1;而 smoke 评分记录是 00321/00564/00385。可达增益是样本特异的,阈值不可跨记录搬运。

**F4 — 候选在 uniform 上确有真实退化,不只是口径。** correction_energy_ratio 仅 7.5e-4,却把全局 rel L2 从 0.07599 抬到 0.08162(+7.4%);该修正能量约占误差能量 13%,方向失配即净负。故需要方向性(parent-relative)监督,而非仅换口径。

## evidence

- `results/r16_dscp_v14/smoke/terminal.json`:gates.oracle_gain、records[0] time_bands.late 63.5 vs parent 49.2、correction_energy_ratio 7.5e-4、aggregate 0.08162 vs parent 0.07599;latency adapter 0.0234/0.0673、wall 137/600s。
- `saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v3.py:130-150,310-322`(未平方口径)、`r16_dscp_training_v2.py:113-175`(平方口径)。
- `results/r4e7_family_temporal_pod_capacity_train9_v1_20260825.json`:uniform rank16 reduction −1.639 / capture 0.2317;rank32 仍 −1.927(加 rank 不救逐帧口径)。
- `results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json`:layered rank16 capture 0.643、rank32 reduction 0.699。
- 已证伪:`parent_travel_time_warp_oracle_r4_train6`(最优乘子 ≈3.5e-18,improvement 全 0);`parent_time_dilation_oracle`(0.00107);`parent_correction_scale_oracle_r1b`(0.0087,最大 0.027);`onset_residual_meta_r4_layered1_relative_overfit200_r3`(单记录过拟合 164 epoch 仅降 0.0002,要求 0.20);`r5b_feature_meta_parent_residual_disjoint_gate_r33`(1.25e-6);`cpadc_modern_r5b_transfer_r4_audit`(digest 不匹配,拒绝直接迁移);`capacity_physical_limit_study.md`(高频头 <1% 能量、空间模态无余量)。

## 回答

**1. 目标函数(直接优化 parent-relative 增益)。** 令 r_t(θ)、r_t^par 为同一未平方逐帧相对 L2,分母加**记录级能量下限** d_t = max(‖truth_t‖², τ·max_s‖truth_s‖²),τ=1e-3。定义可部署代理的帧掩码 m_t(由 **parent 自身帧能量**决定,推理期可算、不需真值):m_t = 1[E^par_t ≥ τ·max_s E^par_s]。损失

L = Σ_family (1/3) · [ Σ_t m_t r_t / Σ_t m_t r_t^par ] + λ_h · mean_t m_t·relu(r_t/r_t^par − 1)² + λ_c·‖c‖²

第一项是门的**同一比值**(比值-of-均值,parent 为常数,梯度即 Σ∂r_t);第二项是逐帧"不劣于 parent"的 hinge,消除"赢一帧输十帧";族间取等权(或 min-over-family)使 uniform 不被交易掉。λ_h≈0.5,λ_c 沿用 1e-4。**不改 `deployment_features` / 参数量 / forward 通道**,仅换损失 + 一个逐帧标量乘,部署三门不受影响。

**2. 族间符号。** 不做按族门控、不做按族新基(1202 参数下无预算,且会把 router 变成精度依赖项)。做**能量区段门控 + 弃权**:m_t 帧掩码(物理依据:均匀介质波前经 PML 离场后帧能量→数值噪声)+ 记录级弃权(router abstain、basis_condition>1e3、有效帧数 <R_min=32、‖c‖ 超族尺度上限)时输出**逐比特 parent**。族仅保留现有 basis 路由。

**3. train-only 容量测试(最小代价证伪"rank16 不足")。** 三腿,单 GPU 合计 <15 min(先例:r4e7 9 记录 117 s):
- A 口径重算:在 smoke 同三记录上按新掩码口径重算 rank 8/16/32 oracle 增益。若 rank16 ≥ 0.20 而 rank32 增量 <0.05 → rank 非瓶颈。
- B 单记录过拟合:1202 参数网在**单条**记录上跑到收敛,比其增益与该记录 oracle;≥0.9×oracle → 容量充分(证伪不足假设);<0.5×oracle 且 loss 已平 → 映射容量不足。
- C 特征腿:用已存在的 480 参数 `RidgePointwiseBaseline` 闭式拟合加权 LS 系数目标,比较 R² 与 B。ridge≈net → 瓶颈在 29 通道特征,不在参数数。

**4. 不要再提:** parent travel-time warp、全局时间膨胀、标量 parent-correction 重标定、CNN 残差 meta-adapter(r4 parent)、onset 两帧误差驱动的整段未来修正、CPADC r5b→r4 直接迁移、高频空间头、加空间谱模态/rank。

## uncertainty

未读 `r16_dscp_v1..v13` 的 design_preflight 全部内容,不排除新口径曾被讨论过;smoke 每族仅 1 条记录,F4 的"方向失配"结论样本量为 1;m_t 用 parent 能量代理 truth 能量的保真度未测(需 A 腿附带报告 mask 一致率)。

## recommended_next_step

冻结 v15 预注册:新口径定义 + 三记录 oracle 常数**在同一记录上重算**(不得沿用 v14 硬编码常数)+ A/B/C 三腿;A 腿先跑(<3 min)。接受:B 腿 ≥0.9×oracle。回滚:v15 smoke 若 adapter p95 >0.10 s 或 e2e/parent >1.02(内缩阈值,留 0.50/1.05 余量)即回 v14 代码路径,保留 v14 best/last。**确认必须在与 96 条 basis 排除集及 smoke 面板 group-disjoint 的 train 记录上、每族 ≥2 条**复现,否则不进 pilot。

## veto_reason

否决三件事:(a) 任何把 oracle_gain 阈值/常数往 v14 实测值方向调的做法(用结果调门);(b) 任何需要部署期真值(含真值帧能量)的掩码或门控;(c) 沿用当前跨记录、含合成样本的 oracle 常数作为新口径的验收基准——必须在被评分记录自身上重算。
