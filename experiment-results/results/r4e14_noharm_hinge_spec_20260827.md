# r4e14 探针规格:no-harm hinge 与长训地平线(2026-08-27)

## 背景与问题

v16 long_ddp4(terminal `results/r16_dscp_v16/long_ddp4/terminal.json`,fail_gate)
joint +0.0203 过门、三族 per-family 全过,唯 nonworse 15/24 败。9 条 worse 全部是
微损(gain_iii ≥ −0.0059,绝对恶化 ≤ 0.006 rel_l2),其中 6 条 uniform 好 parent。
训练损失只拟合场误差,不含任何"不得劣于 parent"的项;patience=4 在 epoch 10 早停
(447s / 7200s 预算),monitor(校准集平均损失)在 ~0.378 平台。

lead 指令(2026-08-27):最终精度是长训的结果,而非几步就到。

两个 v17 前置问题,本探针一次回答:

- **Q1(hinge)**:训练损失加可微 no-harm hinge(惩罚 cand_rel > par_rel 的记录)
  能否降低 held-out 危害数/危害幅度,而不牺牲 joint 增益?
- **Q2(长训)**:同一配方在 12288 upd(v16 long 实跑 15360 全局 upd 的同量级、
  v16 pilot 3072 upd 的 4 倍)地平线上,held-out 增益轨迹是否仍在爬升
  (支持 v17 加大 patience/max_epochs)?

## 数据与角色(全部 train 侧,零新开真值)

- 数据源:/dev/shm/v16_long_bundles/shard_{0,1,2}.pt(v16 long 已建的 216 bundle,
  含 192 long_fit + 24 long_calibration)。本探针**只使用 192 long_fit 记录**,
  不触碰 long_calibration(留给 v17 正式 confirm)、不触碰 pilot_confirm、
  不开任何 sealed split。
- 划分:192 条按族分组、族内按 sample_id 字典序每第 4 条(index%4==3)取为
  holdout(48 条,族均衡 16/16/16),其余 144 条为 fit_probe。规则确定性,无随机。

## 臂(3 臂,同 seed=372,同预算,唯一自由度 = hinge 配置)

结构/优化器/lr 逐字沿用 v16 long:Wide128Head(25266 参)、AdamW lr 3e-3 恒定、
betas (0.9,0.99)、wd 1e-4、clip 1.0、field 单目标(masked_confined_loss 总项)、
冷启动、单 GPU、每 epoch 对 144 条 shuffle。

hinge 定义(可微,逐记录):r = rel_l2(adapted_future, truth) / par_rel(常数),
hinge = relu(r − (1 − margin))。总损失 = losses["total"] + λ·hinge。

- A 对照:λ=0
- B hinge:λ=1.0, margin=0
- C hinge+边际:λ=1.0, margin=0.02(要求至少 2% 相对改善,否则受罚,
  避免 hinge 在拟合记录上迅速失活)

预算:每臂 12288 upd,每 2048 upd 在 holdout 48 上评一次(6 个快照),
无早停,固定预算跑满。评分用冻结 score_bundle 逐字(gain_iii、nonworse)。

## 判读(咨询性探针,非门控;先验写死)

主判据在 holdout 48 最终快照:

- **E5a(hinge 有效)**:B 或 C 相对 A 危害数(gain_iii<0 计数)降 ≥30%,
  且 joint mean gain 不低于 A 的 80% → v17 纳入该 hinge 配置。
- **E5b(hinge 无效)**:危害数降 <30% 或无差别 → v17 不加 hinge,
  依赖容差重议 + 长训预算。
- **E5c(hinge 有害)**:B/C joint 低于 A 的 80% → 弃 hinge,同 E5b 路径。

Q2 判据:A 臂 holdout joint gain 轨迹最后 1/3(快照 4→6)仍上升 ≥0.002
→ v17 加大预算有据;否则平台,v17 预算按 v16 量级即可,精度靠结构项。

## 资源与卫生

- 单 GPU cuda:0,VRAM 预计 <3 GiB;单进程加载 3 shard ≈28 GB 常驻内存(远低于
  240 GiB cgroup 上限,lead 已指示不在内存问题上花时间)。
- 预计总时长 ~35-45 min(3 臂 × (~8-10 min 训 + ~1 min 评))。
- 产物:results/r4e14_noharm_hinge_20260827/terminal.json(+ 各臂轨迹),
  报告 results/r4e14_noharm_hinge_report_20260827.md。不写入任何冻结链目录,
  不改动 parent/basis/panels。
- 运行需 CUBLAS_WORKSPACE_CONFIG=:4096:8(determinism)。
- 本探针不解锁任何 v16 门;v16 记账保持 fail_gate 不变。v17 阈值属新预注册,
  本探针结果作为披露的先验依据,不构成把 v16 阈值向实测值移动。
