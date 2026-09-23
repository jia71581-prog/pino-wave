# L1 重锚门预注册 v3:事后 checkpoint 网格测量(取代 v2 飞行门控)

状态:REVISION v3。基文本 = `research/marmousi_longtime_levers_20260912/L1_GATE_PREREGISTRATION_DRAFT.md`(v2,auditor 有条件通过,R1-R8 已修订)。
日期:2026-09-13(UTC)。
性质:文档修订。本文件不启动任何 GPU 工作,不改动任何训练代码;所述长训已在执行中。
依据证据:`research/marmousi_longtime_levers_20260912/DIAGNOSIS_AND_LEVERS.md`(不变)。

## 0. 偏离记录(必须先读,不做粉饰)

已执行的运行与 v2 设计不一致。以下逐条是实测事实,不是事后重述为原计划:

1. **fork 点不同。** v2 §2 写"父:step 22814 checkpoint"。实际 `fork_audit.json` 记录 `event: fork_partial`、`global_step: 3740`、`epoch: 20`、冷 Adam(`optimizer: fresh`,`optimizer_state_entries_at_start: 0`)。
2. **终点不同。** v2 写固定终点 41,514(22,814+18,700)。实际 `max_steps = 22814`,`start_step = 3740`,`additional_updates = 19074`。
3. **飞行门控从未发生。** 运行以 `--train-only` 执行(`run_identity.json: train_only=true`,`validation_labels_accessed=false`,`test_id_labels_accessed=false`)。日志事件只有 `train_step`、`train_only_epoch`、`resume`、`resume_lr_assert`;没有任何 validation 或误差指标。因此 v2 §4 的 G1 epoch-20 杀停判定、R4"超时视同 fail 杀停"**都没有发生,也不能追溯发生**。任何声称做过 G1 止损决策的表述都是伪造。
4. **v2 的门步号全部失效。** v2 的 G1 "epoch 20 = step 26,554"、超时线 28,424、G2 复测 34,034 都建立在 fork-from-22814 的假设上,对真实 schedule 无意义。

**v3 的定位**:把三个门从**飞行中杀停规则**改写为**在已保留的逐 epoch checkpoint 网格上的事后测量**。杀停语义作废(训练已在途,且无门可杀);门定义本身(度量、清单、阈值、比值)在**种类上不变**。v2 §5(与既有证据的关系)与 §4 的 R6 归因边界原文保留,见本文 §5、§6。

## 0.1 谱系不匹配(第二处偏离,比步号更重要)

v2 假定 A4 与 L1 共享 step 22814 的单一父模型。实际两臂 fork 自**不同 checkpoint、不同 parent config digest、不同增量长度**:

| | A4 基线 | L1 锚臂 |
|---|---|---|
| 产物 | `a4_fulltrain100_20260912_v1/attempt_001` | `l1_anchor_fulltrain_20260913_v1/attempt_003` |
| parent 路径 | `a4_longrun_20260911_v1/training/radial44/checkpoints/checkpoint_step_00004114.pt` | `<repo>/checkpoints/ic8_fullfam_parent_e20_best.pt` |
| parent sha256 | `f88ad285b34181f11c81ae60cf0ab82a31faa8cf9ad9ae16b36e9aaf4f92879d` | `4aa9c47d542fd1e7cb25a3746497da59c965056588d530c771c38a9aaa066b8e` |
| parent config digest | `0e3348c3515d8169531ac94d0712f24a8734caa4564863c70ab8714386fc59c0` | `0ab983a59ff858dc864378e929c07c35be21ce231fa22213a24ad8cf3081793f` |
| fork step | 4114(epoch 22,warm Adam,155 态) | 3740(epoch 20,冷 Adam) |
| additional_updates | 18,700 | 19,074 |
| 终点 | 22,814 | 22,814 |

终点相同是 step 上限的巧合,**不是预算配平**。因此:

- **A4 与 L1 之间的差值不是单变量比较。** 它把 `data.anchor_fraction 0.0 -> 0.5` 这个杠杆与不同 fork origin 混淆,不能单独归因于锚契约。
- A4 的 checkpoint 网格跨 4301..22814;`/root/autodl-tmp` 下不存在任何 step-3740 的 A4 checkpoint。L1 的 fork 父只存在于 repo `checkpoints/` 目录。
- 配置层面确实只有一处杠杆差(`anchor_fraction` 0.0 -> 0.5,加 checkpoint_dir 与注释),已核 diff;但配置单变量不等于谱系单变量。

## 0.2 两个 parent 角色(绝不可互换)

- `fork_parent`:step 3740,sha `4aa9c47d…a066b8e`。**子模型实际发散自它**,是 step 0 唯一函数保持意义下的比较对象。
- `context_reference`:A4 step 22814,sha `ced7128a9b7a683f55914c5af05d3ffc5a3312d1677146411387f6ccfdb804cd`。现有最好模型,可作背景报告,**必须标注为混淆比较(non-single-variable)**。

## 0.3 门的参考模型不对称(用户决策,已记录理由)

G1 与 G2 **使用不同的参考模型**。这是刻意设计,不是笔误:

- **G1 -> fork_parent(step 3740)**。G1 是机制门,其 0.6x 只有相对于子模型真正发散出来的那个模型才有意义,也是 step 0 唯一函数保持的比较。
- **G2 -> context_reference(A4 step 22814)**。G2 是端到端质量门,标杆必须是现有最好模型,而不是子模型恰好分支自的模型。

推论(必须显式记录):v2 的 0.5107 与由此得到的 0.46 绑定值,溯源为 `/root/autodl-tmp/staging/marmousi_gap_probe_20260912/train_records.jsonl` 96 条均值 0.510744(min 0.229,max 0.7569),其 `PROBE_META.json` 记 checkpoint_sha256 `ced7128a…804cd`、global_step 22814、config_digest `f4c44416…`、manifest_digest `55fbffa9…`。即 0.5107 出自 **A4 step-22814 模型 = context_reference**,不是 fork_parent。v2 只假定一个父模型时"0.9x 复算父均值"无歧义;角色分裂后若把该规则套到 fork_parent(step 3740,早期训练),会导出显著更松的界,使 G2 近乎必过。那是对门的静默削弱,**禁止**。故 G2 的界保持绑定在 A4 step-22814 上,值仍为 v2 冻结的 0.46。

## 1. 契约改动(实测,替代 v2 §1 的预期描述)

真实 fresh(新建)键 6 个,来自 `fork_audit.json: fresh_tensors`:

- `source_encoder.anchor_projection.weight` / `.bias`
- `coordinate_encoder.anchor_raw_projection.weight` / `.bias`
- `dense_decoder.blocks.0.spectral.weight`
- `dense_decoder.blocks.1.spectral.weight`

`parent_only_tensors` 4 个:`dense_decoder.blocks.{0,1}.spectral.weight_{top,bottom}`。继承键 153 个,逐张量 bitwise 核验通过;`reset_dense_spectral=false`;`lr_before=lr_after=[2e-4]`。

**函数保持只是部分成立(v2 §1.4 的整体性表述被实测否定)**:两个 anchor 投影确实零初始化(`features.py:151-153`、`source.py:37-39`),但两个 `spectral.weight` 是 `nn.init.uniform_(-scale, scale)`(`spectral.py:125-129`,`scale=1/sqrt(in*out)`),**非零**,且与 A4-radial44 同名 fresh 张量的 sha256 **不相等**。所以 fork 时子模型对父模型**不是逐位一致**,v2 的 G3(max abs diff ≤ 1e-6)在真实运行上不成立、也未被执行。这个偏离影响 §0.2 中"step 0 函数保持"的强度:相对 fork_parent 它是**最接近**函数保持的比较,但不是严格的。

数据侧实测:`anchor_fraction=0.5`;锚上界 = `length - ic_frames - 100`,再与逐记录能量上限取小(`_ANCHOR_ENERGY_FLOOR=0.03`,`pilot.py:490-513`),即记忆中记录的"锚范围 377->293 + coda 能量守卫"。

## 2. 真实 schedule 与步号重映射

`steps_per_epoch = 187`;fork 3740(epoch 20)-> 终点 22814;跨度 19,074 步 = 102 epoch。v2 的 epoch 标号相对 fork 之后计数,重映射公式:`step = 3740 + 187 * epoch_after_fork`。

| v2 门 | v2 步号(基于 fork@22814,失效) | v3 真实步号 | epoch(绝对) | checkpoint 状态 |
|---|---|---|---|---|
| fork 点 | 22,814 | **3,740** | 20 | repo `ic8_fullfam_parent_e20_best.pt` |
| G1 首测(fork 后 20 ep) | 26,554 | **7,480** | 40 | 存在(attempt_003) |
| 门超时线(fork 后 30 ep) | 28,424 | **9,350** | 50 | 存在(attempt_003);超时语义作废 |
| G2 复测(fork 后 60 ep) | 34,034 | **14,960** | 80 | 存在(attempt_003_resume12903) |
| 终点 | 41,514 | **22,814** | 122 | 训练在途,尚未落盘 |

其余可用网格:attempt_003 保留 49 个 checkpoint(3927..12903),attempt_003_resume12903 已保留 24 个(13090..17391,随训练增长)。逐 epoch 保留,故门可按 epoch 轨迹报告。

## 2.1 中断/续跑与重叠(分析必须遵守)

容器重启在 step ~13008 打断,自 `attempt_003/latest.pt` 于 step 12903 续跑进第二产物目录 `attempt_003_resume12903`;五个 digest(code/config/manifest/normalizer/schedule)已核验一致(`resume` 事件 `global_step: 12903, epoch: 69`;`resume_lr_assert` lr 2e-4 未变)。

- attempt_003 的 `metrics.jsonl` 覆盖 3741..13008;resume 目录覆盖 12904..(在增)。**12904-13008 区间两目录重叠**(那 105 步被重跑)。
- 规则:该区间**一律取 resume 目录**;attempt_003 的 12904-13008 记录视为被弃置的尾部,不得混入任何均值、趋势或损失曲线。
- checkpoint 侧无歧义:12903 及之前只在 attempt_003,13090 及之后只在 resume 目录。

## 3. 测量(train 记录;只读 checkpoint;沿用 v2 §3 协议)

固定清单:`research/marmousi_longtime_levers_20260912/MEASUREMENT_LISTS_FROZEN.json`,SHA256 `053d64026450059292554b02329aa5c665f53d1031c96aee5647753724f2cd91`(M 96 marmousi + N 32 uniform + 32 layered,`frozen_utc 2026-09-12T13:43:12Z`)。

测量项 M1 / M2-M3 / N1 与 v2 §3 逐字相同(锚 `a ∈ {onset, onset+100, onset+200}`,各预测其后 100 帧,窗内相对 L2;组合推理 oracle 与 composed 两变体 + 放大系数 `γ_k`;N1 完整未来相对 L2)。执行约束同 v2:单卡 `cuda:0`、`time_block=1`、逐条记录、仅持久化逐 record 指标 JSON(<50 MB)、不存原始波场(R7)。**长训占满 4 卡至今日约 20:01,任何测量必须在其后或在确认空闲显存 ≥8 GiB 时执行,且不得影响训练进程。**

红线不变:validation / test_id 零读取;A4 与 L1 产物目录对本测量只读;新产物只写 `research/l1_gate_measure_20260913/` 与(如需)独立 staging 目录。

## 4. 门定义(种类不变,重锚到真实步号)

- **G1(机制门)**:M1 中 `a=onset+200` 的 **oracle** 重锚窗误差均值 ≤ **0.6 x** 冻结 **fork_parent**(step 3740,sha `4aa9c47d…`)同绝对帧窗均值。
- **G2(端到端门)**:composed 完整未来 train-marmousi 均值 ≤ **G2 绑定值**,且 N1 非退化:uniform/layered ≤ 父+0.01,marmousi onset one-shot ≤ 父+0.02。
- **G2 绑定值规则(v2 §3 R2 原样保留,参考模型钉死为 context_reference)**:若同脚本复算 **A4 step-22814** 的 M-list marmousi 均值 ∈ **0.5107 ± 0.002**,则界冻结为 **0.46**(明示 0.46 比严格 0.9x0.51074=0.4597 略松,为绑定值);否则在**读取任何子模型数值之前**按 0.9x 复算值重冻。读到子模型数值后任何方向的界移动 = 否决。
- **复算兼作跨脚本一致性检查**:现有 96 条探针由 `research/a4_final_validation_20260912/train_marmousi_probe.py` 产出,与本轮新测量脚本不同。若复算落在 ±0.002 窗外,那是**关于两脚本是否一致的证据**,不是在看过子模型数字后移动界的许可。
- **G3(函数保持)**:v2 形式(max abs diff ≤ 1e-6)在本运行**不适用**,因随机初始化的 dense 谱核(§1)。作废,不替换。
- **杀停规则(v2 §4 G1 败杀停、R4 超时视同 fail)**:作废。事后测量无法杀停已在途的训练。

## 4.1 取代飞行门控的排序纪律

1. 先用最终测量脚本测 **fork_parent(step 3740)** 的 M1 各窗,与 **context_reference(A4 22814)** 的 M-list / N-list N1,逐 record 数值 + 均值 + **脚本 SHA256** 写入冻结记录。
2. 在该时刻按 §4 规则**定下 G2 界**。
3. **只有在 1-2 完成并冻结之后**,才允许读取任何子模型 checkpoint 的测量值。
4. checkpoint 网格允许把 G1/G2 作为**沿 epoch 的轨迹**报告,这比 v2 的三次飞行测量信息严格更多。**轨迹必须完整报告**(全部已测 epoch),禁止只报某个挑选出来的 epoch。若轨迹非单调,报告非单调本身,不取最优点代表结论。

## 5. 与既有证据的关系(v2 §5 原文保留)

- B2 快照 IC v2(20260830,旧架构):快照锚训练双门 PASS 但一次性推理"尾部未动"。本实验的新内容是**组合推理协议**与放大系数测量;若 G1 败,与 B2 结论合并为"重锚方向整体证伪"。
- R201 / 损失加权 / 带宽 / 泛化四排除见诊断文档;本设计不动采样机制、不动 loss、不扩带宽、不加数据,单变量 = 锚契约(配置层面成立;谱系层面见 §0.1)。

## 6. 许可与不许可的 claim

- 本实验仍是**先决/机制实验**。无 promotion,无候选(candidate)身份,无推广(generalization)声明。产出的 checkpoint 不参与任何 promotion。
- **G2 归因边界(v2 R6 原文)**:G2 通过支持的 claim 是"锚契约 + 其自带的窗口增广"作为整体优于父模型,不支持把改善孤立归因于契约本身;候选阶段如需归因,另行设计 `anchor_fraction` 消融。G1 为机制门,归因干净(相对 fork_parent)。
- **额外边界(v3 新增)**:任何 A4-vs-L1 的差值**不得**作为锚杠杆的单变量证据(§0.1)。可作背景报告,须显式标注 confounded。
- 无 validation / test 数值进入本实验的任何判定。

## 7. 预算

GPU:零新增训练。测量为单卡推理,父侧基线 + 网格轨迹,按 v2 口径每 checkpoint 分钟级到十分钟级;必须排在长训之后或空闲显存允许时。人工:测量脚本(由并行 agent 实现)+ 本预注册与其 JSON 冻结。

