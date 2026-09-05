## finding

三个矛盾中有两个是**门槛/参照物定义缺陷(target-limited)**,一个是**欠优化(optimization-limited)**。架构与采样不是当前地板。

1. `loss_reduction=0.99967` 是跨记录伪迹,不是拟合成功。smoke 用轮询喂 3 条记录,却把 `losses[0]` 与 `losses[-1]` 当同一目标的首末值。192%3=2 ⇒ initial 取自 records[0](uniform),final 取自 records[2](marmousi)。核验:1−0.5876/1767.36=0.99966752,与门槛值逐位一致。即 0.5876 是 marmousi 记录**近初始**的 loss,与 uniform 的 1767 之比,恰因 uniform 记录晚期 truth 帧能量塌缩(per-frame 分母 `clamp_min(1e-30)`)使其 loss 高 3 个数量级。模型其实几乎没动过。

2. 压制项**不是正则**。`normalized_coefficient_energy` 权重 1e-4,且 `coefficient_scales = 3×` oracle 系数 RMS,不构成上限;tanh 在 0 附近斜率为 1,也不饱和。真实抑制来自:输出层零初始化(correction 恒等于 0 起步,且 `depthwise_hidden` 首步梯度为 0)+ 每条记录仅 64 步 AdamW(lr 3e-3,Adam 每坐标步长 ≤ lr ⇒ ‖ΔW‖∞≤0.19)+ `grad_clip_norm=1.0` 在 loss≈1767 时长期饱和、且 Adam 二阶动量被 uniform 记录主导,压低 layered/marmousi 方向的有效步长。量级证据:oracle rank16 的 correction_energy_ratio 为 uniform 0.0149、layered 0.1139,而候选是 0.00075/0.00045,即幅度差 4.5×/16×——欠训练,不是被夹住。

3. 族间符号不一致来自 **oracle 常数取自另一套记录**,以及 uniform 记录的分母塌缩。常数 0.2225/0.3733/0.2979 出自 `train_uniform_00102`(k1=27,parent per-frame 0.2232)、`train_layered_01032`(k1=14)、`synthetic_train_marmousi_fresh_v1`(合成记录)。而 smoke panel 是 `train_uniform_00321`(parent per-frame **16.92**,late 49.22)、`train_layered_00564`、`train_marmousi_00385`。参照制度与被测制度差两个数量级。uniform 晚期 truth 能量→0,rank16 raw temporal POD 在该尾段无分辨力,任何非零系数都在 truth≈0 处注入能量 ⇒ late 63.52 vs 49.22、mid 2.29 vs 1.17;layered/marmousi 晚期 truth 未塌缩,同样微小修正即给出正增益(0.8977 vs 0.9984)。`condition=1.0015` 只查数值条件数,查不出能量错配。该单条记录的 +0.0056 也正是 `aggregate_nonworse` 以 0.7836 vs 0.7825 落败的全部来源。

## evidence

- `saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v6.py:88`(`run_v6_smoke`:`records[updates%len(records)]` 后 `smoke_gates(losses[0],losses[-1],…)`)
- `…/r16_dscp_engine_v3.py:320-321`(硬编码 oracle 与 `reduction` 公式)、`:310-317`(`_family_gain` 用 `mean_frame_rel_l2`)、`:136-137`(per-frame 分母 clamp)
- `…/r16_dscp.py:285-286`(输出层零初始化)、`:297`(tanh)、`:329`(×scales)、`:328`(`CONDITION_MAXIMUM` 仅数值条件)
- `scripts/train_r16_dscp.py:646-648`(scales=3×RMS)
- `…/r16_dscp_training_v2.py:113-172`(loss)、`:176-207`(`weighted_coefficient_target`,与训练 loss 同权,故范数无失配)
- `…/r16_dscp_engine_v5.py:78-95`(`_forward_loss` 传入真系数;clip 1.0)
- `results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json:25-42, 104, 320-354`(oracle 出处、样本 id、k1、能量比)
- `configs/r16_dscp_v14.yaml:12-13, 23, 25`
- `results/r16_dscp_v14/smoke/terminal.json:20,49-62,121-123,124-232`

**accepted vs rejected**:smoke 无 epoch 接受/拒绝机制;192/192 更新全部接受,`early_stop=false`,cache 420 hit/3 miss,无一次 `NonFiniteStepRejection`/`OOMStepRejection`(会直接抛出并终止,engine_v5:80-100)。patience/best 逻辑属 long 阶段,本次未运行。故"拒绝重试"数为 0,不能用来解释地板。

## uncertainty

沙箱拒绝执行 python,未能直接读取 `smoke/last.pt` 的 `pointwise_out` 权重范数,"欠训练"依赖 correction_energy_ratio 代理量。smoke 三条记录的 k1 未在 terminal 中暴露,uniform 分母塌缩的程度只能由 late 49.22 反推。每族仅 1 条记录,`-0.3052` 无方差估计。

## recommended_next_step

单次 oracle 上界探针,零训练、约 2 分钟、1 GPU:对 **smoke panel 这 3 条记录**调用 `weighted_coefficient_target(bases[route], parent, full_truth, k1)`,套 `c1_causal_mask` 后用 `metric_record` 打分,输出每族 `mean_frame_rel_l2` 增益与 `correction_energy_ratio`(复用现成 `ridge_update/ridge_score` 通路,仅换系数来源;结果只打印到 stdout)。判定:若 uniform 的 oracle 增益 ≥0.111 且 layered ≥0.187、marmousi ≥0.149,则基与度量无碍,地板是优化(下一步扩更新数 + 逐族 loss 归一化);若 uniform oracle 增益 ≤0 或 ≪0.111,则门槛在本 panel 上不可达,先修 panel/度量再谈 long。此单值即可二分。

## veto_reason

禁止以本次 smoke 证据晋级 pilot/long,并禁止对外表述"拟合目标被压到极致":`loss` 门槛的实现无法度量优化进展(任何 records[0] loss 远大于 records[2] 的候选都能通过),该表述违反"不得以训练损失宣称目标"。同时 `oracle_gain` 门槛在参照物修正前不可判定——参照 3 条记录与被测 3 条记录不同,其中 marmousi 参照还是合成记录。修正门槛定义(同 panel oracle、或 per-record 归一)之前,不应消耗任何长跑预算。
