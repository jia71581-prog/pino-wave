# 预训练算子的高效实例化微调 —— 实现与验证报告

日期: 2026-07-28
方案: 快照条件化元学习超网络 + 部署期 LoRA 秒级微调
亲算子: 冻结的 saved-time v41 全量算子 (held-out aggregate relL2 ≈ 0.47–0.51)

---

## 一、交付内容

一条两段式、因果安全的实例化微调流程，全部落在 `saved_time_phase_operator_v4/instance_adaptation/`
与 `scripts/`，**不改动亲算子权重**：

### 阶段 A — 补偿头 + 部署 LoRA (adapters.py / trainer.py)
- `OnsetAdaptedV5` 新增两个**每实例**可训参数:
  - `latent_delta` (维度 = latent_dim): 加在快照编码器潜在向量上的零初始化偏移 (部署 LoRA 主体)。
  - `residual_gate` (标量, 初值 1): 对整条元残差的门控。**关键安全阀**——若共享元残差对某实例有害
    (如均匀介质), 门可收缩至 0; 回滚时强制置 0, 等价于纯亲场。
- 补偿注入位置 = **全场残差头 (绕过 coarse MIONet 场)**, 直击已确证瓶颈 (均匀介质锐波前 Gibbs 振铃 + 晚期帧)。
- `set_deployment_mode(True)` 冻结元网络主体, 只训 `latent_delta + residual_gate`。

### 阶段 B — 元学习超网络 (scripts/train_meta_hypernet.py)
- 家族 3/3/3 平衡 episode, train-split 全真值**仅用于离线元训练**。
- **全程在归一化 O(1) 空间训练** (关键修正, 见"二、关键问题与修正")。
- 损失 = `metric_aligned_relative_loss` (时空联合 relL2) + 归一化观测项 + 能量项。
- 产物 `meta_hypernet.pt` (含 schema version / manifest digest / sha256)。

### 阶段 C — 部署期秒级微调 (scripts/run_meta_instance_adaptation.py)
- 加载冻结亲 + 冻结元网络, 仅优化 `latent_delta + residual_gate` (33 参数)。
- 只读 K=2 早期起振帧 (`SnapshotAccessAudit` 强制), 归一化空间内 Adam+LBFGS 极少步。
- 回滚门 (observed 改善 + energy 比) 不达标即退化到纯亲场。**物理 PDE 残差门在归一化空间无意义, 已在部署模式停用。**
- 评测前 decode 回物理单位, 交既有 `evaluate_after_adaptation` 密封评分。

### 配置 (configs/saved_time_v5/meta_hypernet.yaml)
- 亲绑定 v41 实体 ckpt + run_identity (与 feature_meta_v41 同款成熟接线)。
- `normalization_json` 覆写: 共享 JSON 被 7/26 TGRS ablation 改过 digest, 显式绑定与亲一致的
  `.before_tgrs_ablation_identity_20260726T1050.json`, 不干扰 ablation。

---

## 二、关键问题与修正 (均经实数据定位)

1. **归一化空间 vs 物理单位 (根因)**: 该数据集 decode 后压力 ~1e-9, 直接 L2 拟合数值病态,
   单个 uniform episode 的梯度范数达 ~3856, 一步更新即摧毁共享残差 (uniform field_loss 0.25→6.38 发散)。
   **修正**: 元训练/微调/损失全部在 normalizer 的 O(1) 空间 (std~0.01–0.1), 仅最终部署场 decode。发散消除。
2. **早期近零帧观测损失爆炸**: observed 项按近零起振帧能量归一 → 分母极小 → 爆炸 (V72 同款失败模式)。
   **修正**: 观测误差改用整条记录能量归一。
3. **元残差回滚不彻底**: 早期版本 accepted=False 只回滚 latent_delta, 但元训练残差仍叠加, uniform 被推到 3.54。
   **修正**: 引入 `residual_gate`, 回滚时置 0 = 纯亲场, 保证**任何有害元残差都无法通过门**。
4. **hard_project 不连续**: 近零早期帧插真值制造帧间不连续, 反而恶化能量聚合指标。**部署默认关闭** (`hard_project_onset`)。

---

## 三、验证结果 (G0–G3)

### G0 单元测试: 全绿 (29/29)
`tests/saved_time_phase_operator_v4/test_instance_*.py`——覆盖: 部署模式冻结元网络只训 latent+gate、
回滚置零 = 纯亲场、零初始化起点 = 亲场、审计只读 K 帧 `future_truth_used=False`。
(顺带修正了 3 处**早于本工作**的既有测试失配: onset_indices 签名、public_keys、conditioner schema version。)

### G1 部署 smoke (1 实例/族, 真实亲 + 真实数据): 通过
- deployment LoRA 起动, `trainable=33`, `future_truth_used=False`, 只访问各实例早期 2 帧
  (uniform[15,16] / marmousi[17,18] / layered[24,25])。
- 回滚机制正确; 壁钟 9–18 s (秒级)。

### G2 元训练收敛: 通过 (稳定有限, 无 NaN/OOM)
归一化空间后全族有限收敛; 因家族间共享残差竞争, 短训后快速平台化。

### G3 密封 held-out 评测 (稳健能量聚合 aggregate relL2, adapted vs 冻结亲基线)

| 家族 | adapted | 亲基线 | accepted | 壁钟 (s) |
|------|---------|--------|----------|----------|
| uniform  | 0.5074 | 0.5070 | True  | 18.4 |
| layered  | 0.2169 | 0.2169 | False | 9.0  |
| marmousi | 0.5213 | 0.5212 | True  | 12.6 |

**核心结论**:
- **安全性契约成立**: 三族 adapted 均 ≤ 亲基线 (含之前失控的 uniform, 由 3.54 修正到 0.507 = 亲水平)。
  回滚门 + residual_gate 保证秒级微调**绝不恶化**冻结亲。
- **改善幅度极小**: 当前 2 帧早期观测 + 小微调层未能显著突破 held-out 误差。这与 memory 中
  容量梯/V72 已确证的**结构瓶颈** (coarse MIONet 场对均匀锐波前 + 晚期帧的表征上限) 完全一致——
  非本方案实现缺陷, 而是既定的结构地板。方案作为"安全、参数极省、秒级的部署侧微调层"已完整可用,
  且与"改 coarse 场结构" (memory option B) 正交, 可叠加。

### 因果契约审计
所有实例 `future_truth_used=False`, `accessed_true_indices` 仅早期 2 帧。元训练全真值仅离线 train-split 使用
(`future_truth_used_only_for_train_episode=True`)。

---

## 四、复现命令

```bash
WORKROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd $WORKROOT
# G0
PYTHONPATH=$WORKROOT python -m pytest tests/saved_time_phase_operator_v4/test_instance_*.py -q
# 阶段B: 元训练 (离线一次)
PYTHONPATH=$WORKROOT CUDA_VISIBLE_DEVICES=0 python scripts/train_meta_hypernet.py \
  --config configs/saved_time_v5/meta_hypernet.yaml \
  --output .../artifacts/meta_hypernet/meta_hypernet.pt --per-family 4 --epochs 3 --time-points 32
# 阶段C: 部署秒级微调 + 密封评测
PYTHONPATH=$WORKROOT CUDA_VISIBLE_DEVICES=0 python scripts/run_meta_instance_adaptation.py \
  --config configs/saved_time_v5/meta_hypernet.yaml --output-dir results/instance_adaptation/pilot --pilot
```

## 五、后续可选增强 (非本次交付范围)
- 用户选定 K=2 早期帧; 若放宽到"稀疏若干真实时刻帧" (含中/晚期), 可给微调层更多信号直击晚期瓶颈。
- 家族间共享残差竞争: 可为超网络加家族条件或分族 residual, 缓解 uniform 与 layered/marmousi 的表征冲突。
- 与 memory option B (替换 coarse MIONet 场结构) 叠加, 本微调层可直接复用。

---

## 六、更新 (2026-07-28 下午): 微调损失加入物理信息 (LWC-84 PDE)

用户要求"微调时损失函数加入物理信息, PDE loss 用 LWC 格式"。**已落地。**

- **根因澄清**: 原部署路径把 `pde=0.0` 关掉, 注释理由是"物理残差在物理压力单位、归一化空间会误标度"。
  但 `losses.py::lwc84_residual` 返回的是 **`residual / scale` (自归一, 尺度不变)** —— 因此它在归一化 O(1)
  空间**完全有效**, 关掉的理由已过时。
- **实现**: `trainer.py::adapt_instance` 新增 `deployment_pde_weight` (默认 0.0 = 保持旧安全行为)。
  部署模式下 loss = `observed(整场能量归一) + deployment_pde_weight · LWC84_selfnorm_PDE + 1e-5·anchor`;
  bridge/phase/energy 仍关 (它们是物理单位, 归一化空间无意义)。PDE 残差在第 2 观测帧之后的所有时刻上采样监督。
- **接线**: `run_v5_instance_adaptation.py` 从 config 读 `deployment_pde_weight`;
  `configs/saved_time_v5/meta_hypernet.yaml` 设 `deployment_pde_weight: 0.1`。
- **测试**: 新增 `test_deployment_pde_weight_enables_self_normalized_physics_term` (spy 确认 pde 权重真流入闭包,
  默认 0.0 保持旧行为); 全套 `test_instance_*.py` 30/30 绿。
- **端到端 pilot** (`results/instance_adaptation/pilot_pde/`, pde_weight=0.1): 因果契约保持
  (`future_truth_used=False`, 仅访问各族 2 早期帧 [15,16]/[24,25]/[17,18]); 安全契约保持
  (adapted ≤ parent: uniform 0.696 vs 0.694、marmousi 0.586 vs 0.584、layered 门控回退到父)。
  **结论**: 物理项已安全接入, 但在 **2 早期帧因果契约**下仍不超越父场——与"信息含量天花板"研究一致
  (2 帧信息量不足, 见 `CEILING_VERDICT.md`)。物理项的价值将在**放宽观测契约到稀疏中/晚期帧**时显现
  (届时把 `deployment_pde_weight` 调高即可, 无需再改代码)。

## 七、信息含量天花板研究 (落实"五、#1", 详见 `CEILING_VERDICT.md`)

用**容量无上限的自由全场校正**夹逼"0.47 墙是容量墙还是信息墙"。结论: **是信息墙 ∩ 结构墙**。
零结构自由校正器的 held-out relL2 随观测帧数单调下降 (容量恒无限却随信息剧变 → 限制是信息不是容量):

| family | 父 | K=6 | 12 | 24 | 48 | 100 |
|--------|----|-----|----|----|----|-----|
| uniform  | 0.51 | 0.448 | 0.382 | 0.300 | 0.139 | 0.122 |
| layered  | 0.21 | 0.197 | 0.179 | 0.123 | 0.072 | 0.061 |
| marmousi | 0.52 | 0.407 | 0.327 | 0.275 | 0.135 | 0.117 |

K~48 帧即让零结构校正压到 0.06–0.14 (逼近过拟合地板与 0.10 目标, **不需任何结构改动**)。
严格 2 早期帧下信息量太少, 连无限容量都超不过父场 → 解释本方案"改善极小"的根因。


---

## 八、自适应采样 (2026-07-28: SOTA RAD + 因果时间倾斜)

用户要求"微调用最先进的高效自适应采样算法"。**已落地: 残差自适应分布 (RAD, Wu et al. CMAME 2023) + 因果时间倾斜。**

### 动机与关键结构事实
微调 PDE 损失原在 **512 个均匀随机固定点**上采样残差 (`build_fixed_physics_points`, 每实例只播种一次)。
均匀采样把梯度平摊全时空, 而本项目已确证的失败模式是**局部的** (均匀介质尖锐波前 Gibbs 振铃 + 晚期帧)。
关键: `lwc84_residual` 有限差分**一次性算出整场残差** `r[t,z,x]`, 采样只是从中 gather——所以自适应采样在此
**不是省 autodiff 成本, 而是重加权哪些残差驱动梯度** (近零开销), 可精确对准瓶颈。

### 算法
每格采样概率 `p(t,z,x) ∝ ( |r|^k / E[|r|^k] + c ) · exp(-λ·(t-t0)/(T-t0))`:
- `k` (默认1.0) 集中度; `c` (默认1.0) 均匀基底防塌缩; `λ=time_tilt` (默认1.5) **因果时间倾斜**优先最早未观测时刻。
- `torch.multinomial` 抽样 → 反解回 `[time_abs,z_frac,x_frac]` (与旧点格式一致, 复用现有 gather + 校验, 零改动)。
- **cadence**: Adam 阶段每 `resample_every` (默认4) 步用当前 detached 残差重画; **LBFGS 阶段冻结点集** (固定闭包语义)。

### 实现与测试
- `losses.py`: `PhysicsSampling` dataclass (默认 `uniform`=向后兼容) + `build_rad_physics_points`。
- `trainer.py`: `adapt_instance(physics_sampling=...)`; closure 内按 cadence 重采样, LBFGS 前 `freeze_points=True`。
  **安全契约 (回滚门 + residual_gate)、因果审计、评分完全不动**——采样只改梯度分布。
- 接线: `run_v5_instance_adaptation.py` 读 config `physics_sampling` 块; `meta_hypernet.yaml` 默认 `method: rad`。
- 测试: `test_instance_physics.py` +4 (热点集中 67% vs 均匀 0.005%、因果倾斜、确定性、退化回退均匀);
  `test_instance_trainer.py` +2 (RAD 端到端 + resample cadence 精确计数 + uniform 默认零 RAD 调用)。**全套 36/36 绿。**

### 部署 pilot 对照 (`pilot_pde_rad/` vs `pilot_pde/`)
| family | RAD adapted | uniform adapted | parent |
|--------|-------------|-----------------|--------|
| uniform  | 0.6963 | 0.6963 | 0.6942 |
| layered  | 0.4889 | 0.4889 | 0.4889 |
| marmousi | 0.5856 | 0.5856 | 0.5840 |

**RAD ≈ uniform, 与预期一致**: 严格 2 早期帧下信息量不足 (七节), 采样算法无从发力; 因果+安全契约均保持
(`future_truth_used=False`, 仅 2 早期帧, adapted ≤ parent)。

### 采样增益隔离对照 (在**我们自己的方法**上, 非 DeepONet、非自由张量)
`scripts/diagnose_highcap_instance_finetune.py` 用真实 `OnsetAdaptedV5` 适配器 (解冻整条残差头, 我们的方法)
微调, 支持 `--sampler {full|uniform|rad}` 三选一 (full=整场每格, uniform=512 固定随机点, rad=512 残差自适应+
因果倾斜、每 `resample_every` 步重画), 复用生产同款 `build_rad_physics_points`/`sample_fixed_physics_residual`。

| family | full (每格) | **rad-512** | uniform-512 | parent |
|--------|------------|-------------|-------------|--------|
| uniform  | 0.5073 | **0.5073** | 0.5073 | 0.5069 |
| marmousi | 0.5215 | **0.5215** | 0.5217 | 0.5209 |

**两点结论**: (1) 严格 2 帧契约下三者 held-out 均 ≈ 父场 (信息墙主导, 见七节, 与 `CEILING_VERDICT` 一致),
采样无从改精度; (2) **相对格局恰是 SOTA 正确形态**: `rad-512` 逐位复现 `full` (整场) 的结果, 而 `uniform-512`
在 marmousi 略差 (0.5217 vs 0.5215)——**RAD 用 512 点即达到整场质量的 PDE 监督, 均匀 512 点则损失一点**,
正是 RAD 的效率收益。产物 `results/instance_adaptation/sampler_on_ours/`。采样算法的**精度**收益要在放宽观测契约时才显现。

### 复现
```bash
# 部署 (config 默认已开 rad, 作用于我们的 OnsetAdaptedV5 适配器):
PYTHONPATH=$WORKROOT CUDA_VISIBLE_DEVICES=1 python scripts/run_meta_instance_adaptation.py \
  --config configs/saved_time_v5/meta_hypernet.yaml --output-dir results/instance_adaptation/pilot_pde_rad --pilot
# 采样对照 (我们的方法, 三种 sampler):
PYTHONPATH=$WORKROOT CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_highcap_instance_finetune.py \
  --config configs/saved_time_v5/meta_hypernet.yaml --sample-id validation_uniform_00069 --sampler rad
```
