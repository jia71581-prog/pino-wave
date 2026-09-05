# r4e13 探针规格:跨记录泛化的冷启动归因与数据规模阶梯(冻结于运行前)

日期:2026-08-26。类别:**train-only 诊断探针,非 stage,非晋级候选**。
授权背景:lead"继续"(2026-08-26,v16 pilot fail_gate 报告后)。遵守全部在册 veto;
不动 v16 预注册门;不触 validation/test_id;truth 只开 train split 的 panels 角色记录。

## 问题

v16 pilot fail_gate(joint −0.032,nonworse 10/24,marmousi 首快照即 −0.043)。
两个未分离的解释:

- **H-W(热启动污染)**:pilot 从 smoke last.pt 热启动(实现选择,预注册未写明),
  3 记录记忆污染跨记录学习;
- **H-S(数据不足)**:24 fit 记录不足以学到跨记录一致的特征→系数映射,
  192 记录(long_fit 规模)可能可以;
- **H-F(根本不迁移)**:该映射在 wide128+29 通道下跨记录本质不一致,加数据无用。

## 绑定

v16 runner 的全部 FROZEN 绑定(经 `verify_bindings()` 复核)+ 运行时钉:
`scripts/train_r16_dscp_v16.py`、`scripts/probe_r4e13_crossrecord_scaling.py`、
`results/r16_dscp_v16/pilot/terminal.json`、本规格。

## 共享设定

- 管线:v16 runner 的 `build_bundles`/`bundle_loss`/`score_bundle` 原样复用
  (该路径 smoke 阶段已与 r4e12 W 臂逐位一致)。
- 头:Wide128Head(25266 参),**全臂冷启动**(seed 372,零初始化末层)。
- 优化:AdamW(0.9,0.99)/eps1e-8/wd1e-4/clip1.0,lr 3e-3 恒定,3072 updates/臂,
  shuffled epochs(generator seed 372),每 384 updates 评一次 confirm。
- 评估集:**三臂同一** pilot_confirm 24 记录(held-out train);另在终点评全 fit 集
  (拟合-泛化差读数)。
- truth 允许清单:pilot_fit ∪ long_fit ∪ pilot_confirm(全 train split)。

## 臂定义

| 臂 | fit 集 | fit 数 | 回答 |
|---|---|---|---|
| A | pilot_fit | 24 | 与 pilot 唯一差异=冷启动 → H-W |
| B | pilot_fit + long_fit 前 72(panels 顺序) | 96 | 规模中点 |
| C | long_fit | 192 | H-S:long 规模能否泛化 |

## 预注册判读(confirm 集,3072 终点;g = mean gain_iii)

- **(E1 热启动有害)** g_A ≥ pilot 实测(−0.0323)+ 0.02 → H-W 成立(记录为实现教训);
  若 A 同时过 pilot 三门(joint≥0.01,per-family≥0.005,nonworse≥23/24)→ v16 pilot
  失败主因即热启动,v17 = 冷启动重跑 pilot(新预注册)。
- **(E2 规模有效)** g 随 A→B→C 单调上升,且 C 过 pilot 三门 → H-S 成立,
  v17 = 以 long_fit 全量为 pilot fit 集的新预注册(stage 设计修订并披露)。
- **(E3 不迁移)** g_C < g_A + 0.01 且 C 不过门 → H-F 成立:该架构跨记录不泛化,
  方向转入按族分头/特征条件化路由的新探针(不复活已证伪清单)。
- **(E4 混合)** 其余 → 按臂按族如实报告,基于最大 g 臂选后续。
- 全臂附:fit 集终点得分(拟合-泛化差)、按族分解、marmousi 单列。

## 预算与安全

- 单 GPU(GPU 0),总 wall ≤ 6000 s(估:240 bundle 构建 ~65 min + 3 臂训练 ~5 min
  + 评估 ~2 min);peak reserved ≤ 20 GiB;启动前磁盘 ≥ 2 GiB;CPU RAM ~31 GB(裕量大)。
- 写入白名单:`results/r4e13_crossrecord_scaling_20260826/` 新建;零修改既有文件、
  零删除、零模型检查点落盘。
- 父检查点 sha 前后复算;validation/test_id 不可触;阈值零移动。

## 声明边界

train-only 诊断;confirm 集是 held-out train,不是 validation;E1/E2 成立也只支持
"起草 v17 预注册",不直接晋级;oracle 不在本探针中读取(门用 gain_iii 与 pilot 门
同口径)。已证伪清单不复活。
