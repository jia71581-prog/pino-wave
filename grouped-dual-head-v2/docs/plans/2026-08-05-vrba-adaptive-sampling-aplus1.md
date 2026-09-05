# Plan: vRBA 混合自适应采样接入 A+1(记录级 RAD + 时间帧级 RBA)

## Context

**为什么做这个。** A+1(query-invariant 时间-频域 Helmholtz 合成 + 物理背景 P_bg)已收敛,held-out 0.053。刚上的 **SOAP 二阶优化器续训**已把 aggregate 从 0.0504 降到 0.0411,并把 uniform 晚期近零帧伪影清掉 24×(0.0265→0.0011),但 **layered/marmousi 基本持平**(layered 0.078 / marmousi 0.044)。残留瓶颈是晚期高秩散射残差(render 特征秩塌 8,记忆 §26-27),优化器撼不动,需要在**数据采样**上让高误差的记录 / 时间帧多训。

**当前采样是纯均匀。** g3 脚本 `_train_schedule`(`scripts/diagnose_helmholtz_g3_heldout.py:233`)调用 `build_full_support_schedule` 时没传 `record_weights`/`record_oversample`,即每 epoch 对 192×3 训练池做一次均匀置换,每条记录恰好一次;时间帧维度也无加权。A+1 **从未试过**自适应采样。

**方法选定(用户定,2026-08-05):vRBA 混合(记录级 RAD + 时间帧级 RBA)。** vRBA = variational residual-based attention(Karniadakis 组,Nature NPJ AI 2026),论文明确"时空重要性加权 + 函数实例重要性采样"混合策略,支持 FNO/DeepONet,且推荐配二阶优化器(= 我们的 SOAP,文献背书组合)。官方参考实现已下载 `external/vrba_ref/vrba_sample.py`(863 行 PyTorch)。

**关键区别于已证伪的 late_frame_gain**(记忆 §14 续2/3:硬性给晚期低能帧固定 2× 增益反效果,放大噪声):vRBA 的 RBA 是**有界 EMA 注意力**(`Λ←γΛ+η·λ_it`,上界 η/(1−γ)),按实际残差自适应聚焦,理论上避开 relative-L2 分母趋零病态。

**目标产出:** g3 脚本支持 `--adaptive-sampling {none,rad,rba,vrba}`,在 SOAP 续训基础上跑一个 vRBA run,判据 = 破 SOAP-only 基准(尤其 layered late)。诚实边界:若无改善则记为负结果(晚期是结构瓶颈,采样也救不了)。

## 移植蓝本(已获取的官方实现)

`external/vrba_ref/vrba_sample.py` 两个核心函数:
- `update_spatial_weights(R, Lambda, Par, it)`(line 123):有界权重 EMA `Λ←γΛ+η·λ_it`,`λ_it=φ·(q/q_max)+(1−φ)`,势函数 q 可选(**exponential↔L∞** / **quadratic=r,linear=1↔L²** / lp=r^{p-1} / log-safe 带 Newton 解 ε)。→ 时间帧级(R = 逐帧残差,reduction 沿空间)。
- `update_function_pdf(Lambda_global, Par, it)`(line 204):每样本空间权重求和成标量分数 → 记录采样 PDF。→ 记录级 RAD。
- RBA 原始有界式(参考,已提取):`r_norm=η·|r|/max|r|; λ←γλ+r_norm`(γ=0.999, η=0.001)。

## 关键接入点(已核实)

1. **`build_full_support_schedule`**(`saved_time_phase_operator_v4/full_support.py`)**原生支持** `record_weights`(归一化长度=record_count 向量)+ `record_oversample>1.0`。契约:oversample=1.0 或 weights=None 保持均匀 bit-for-bit;oversample>1 时每 epoch **先铺一次完整均匀置换(coverage 保底)**,再按权重补 `(oversample−1)·N` 次加权抽样。→ 记录级 RAD 直接复用,零新采样代码。
2. **schedule 是一次性预生成**并 pre-bake 进 dataset(`_train_dataset`,line 570)。**架构约束:记录级 RAD 若按残差动态重算,须每个重算周期重建 schedule+dataset+loader。** 简化方案见下(分阶段重建,非逐 update)。
3. **时间帧级注入点**:`_train_update`(`scripts/train_saved_time_v4_full_support.py:1938`)已有 `frame_time_weights` 钩子(现被 `late_frame_gain` 占用,line 48-62)。RBA 帧权重复用这个通路(替换固定 ramp 为有界 EMA)。
4. **per-record 残差来源**:`_evaluate_triplet` 返回 `source_relative_l2`/`medium_relative_l2`(`streaming_metrics.py:234`),但那是 held-out 3 记录。训练池的 per-record 残差需在训练循环里按 sample_id 累积(`_train_update` 能拿到 micro.sample_id + 预测/目标)。
5. **本地已有 RAD 参考** `src/fno_acoustic/train.py::_adaptive_sampling_weights`(EMA + uniform_fraction 混合 + residual_power),范式一致可借鉴,但在 FNO 基线管线,需在 g3 侧重实现。

## 实现步骤

### Phase 1 — 新模块 `saved_time_phase_operator_v4/vrba_sampling.py`(移植 + 单测)
- 移植 `update_spatial_weights` / `update_function_pdf` 的势函数逻辑为纯函数:
  - `vrba_frame_weights(frame_residual[T], lambda_prev[T], *, gamma, eta, phi, potential) -> lambda_new[T]`(有界 EMA,时间帧)。
  - `vrba_record_pdf(record_lambda_scores[N], *, potential, uniform_fraction) -> probs[N]`(记录采样 PDF,归一 + uniform 混合保底)。
- 单测 `tests/tgrs_dclp_no/test_vrba_sampling.py`:①有界性(λ 上界 η/(1−γ));②势函数单调(残差大→权重大);③退化(uniform_fraction=1 → 均匀;eta=0 → 冻结);④PDF 归一 + 非负 + sum=1。

### Phase 2 — 记录级 RAD 接入 g3 脚本
- 训练循环累积每训练记录的 relative-L2 EMA(`residual_ema: dict[sample_id→float]`,在 `_train_update` 后按 micro.sample_id 更新,复用 `train.py::_update_residual_ema` 的 alpha-EMA 范式)。
- 加 `--rad-recompute-epochs K`(默认 0=关):每 K 个 epoch 用 `vrba_record_pdf(EMA 分数)` 生成 `record_weights`,重建 schedule(`build_full_support_schedule(..., record_weights, record_oversample)`)+ dataset + loader。
- 加 `--record-oversample`(默认 1.0)。DDP 注意:所有 rank 用**同一** EMA(需 all-reduce 记录残差)+ 同一 seed 重建 schedule,再 `ddp_update_specs` 切片,保持各 rank disjoint(照现有 line 566 逻辑)。

### Phase 3 — 时间帧级 RBA 接入
- 维护 `frame_lambda[T]`(有界 EMA,per time-bin);`_train_update` 每次用当前 batch 的逐帧残差经 `vrba_frame_weights` 更新,作为 `frame_time_weights` 注入帧损失(替换 late_frame_gain 的固定 ramp 通路,line 48-62)。
- 加 `--frame-rba {off,on}` + `--rba-gamma/--rba-eta/--rba-phi/--rba-potential`。默认 off 保持 SOAP-only bit-for-bit。
- **诚实防护**:RBA 帧权重作用在 per-frame 归一的帧损失上时仍可能触发 §14 的低能帧病态 → 默认势函数用 `quadratic`(↔L²,温和),`exponential`(↔L∞,激进攻晚期)作为可选;记录 layered late 是否被病态放大。

### Phase 4 — vRBA 混合 run + 对照
- `--adaptive-sampling vrba` = RAD(记录)+ RBA(帧)同开。
- 4 卡 DDP 从 **SOAP-only 收敛 checkpoint** 续训(接力,非从 AdamW 起点),与 SOAP-only 单变量对照。
- run 目录 `results/helmholtz_g3_aplus1_soap_vrba_ep40/`;判据 = held-out agg / layered late 破 SOAP-only。
- 消融:rad-only / rba-only / vrba,判断各自贡献(小规模,各 ~40 epoch)。

## 验证
- `pytest -q tests/tgrs_dclp_no/test_vrba_sampling.py`(Phase 1 单测全绿)。
- 退化等价性:`--adaptive-sampling none` 或 `--record-oversample 1.0 --frame-rba off` 与当前 SOAP run **逐值一致**(bit-for-bit,证明零默认改动)。
- smoke:`--smoke-updates 2` 单卡跑通 RAD 重建 + RBA 帧权重(warmstart SOAP ckpt,峰值显存 <23.5GB)。
- DDP 一致性:确认所有 rank 的 record_weights / schedule 一致(同 EMA all-reduce + 同 seed),per-rank 切片仍 disjoint 覆盖。
- 诚实记录:若 layered late 未破 SOAP-only 或被 RBA 病态放大,记为负结果(采样救不了结构瓶颈),不强行调参凑数。

## 关键文件
- 新建:`saved_time_phase_operator_v4/vrba_sampling.py`、`tests/tgrs_dclp_no/test_vrba_sampling.py`。
- 编辑:`scripts/diagnose_helmholtz_g3_heldout.py`(RAD 重建循环 + CLI + residual EMA all-reduce)、`scripts/train_saved_time_v4_full_support.py::_train_update`(frame_time_weights 接 RBA)。
- 复用:`saved_time_phase_operator_v4/full_support.py::build_full_support_schedule`(record_weights/oversample,已支持)、`src/fno_acoustic/train.py::{_adaptive_sampling_weights,_update_residual_ema}`(RAD 范式参考)。
- 蓝本:`external/vrba_ref/vrba_sample.py`(vRBA 官方 PyTorch)。
- 起点 checkpoint:SOAP-only 收敛的 `results/helmholtz_g3_aplus1_r8_sigma2_N192_ddp4_soap_ep80/latest.pt`(待其跑完);cache `/dev/shm/g3cache/background_pbg_sigma2_g3pool_N192.h5`;pinned norm JSON `/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json`。

## 风险 / 诚实边界
- 记忆已证晚期瓶颈是 **render 特征秩塌 8**(结构),采样让高误差样本多训**可能仍撼不动**结构上限 → vRBA 的现实预期是记录级 RAD 帮 layered(欠训成分)、时间帧 RBA 攻晚期(可能触病态)。若整体不破 SOAP-only,这本身是有价值的负结果(佐证瓶颈非采样不足)。
- late_frame_gain 已证伪的病态是主要风险 → RBA 有界 + 默认 L² 势 + 退化防护 + 逐 run 监控 layered late。
