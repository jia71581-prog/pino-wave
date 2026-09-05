# v20 sealed validation 一次性评估报告(2026-08-27)

prereg sha d580afb8(解封 = lead 明示批准"批准解封";评估前冻结,一次性无重试)。
候选 = v18 B_data_hinge best.pt(sha 51c0f0f0)。评估集 = validation 480 条三路由族
(240/150/90),abstain 0,覆盖强制校验过;test_id 保持封存;parent 未动。
terminal: results/r16_dscp_v20_validation_once/terminal.json。

## R1 继承的绝对验收(0.05):FAIL(双形式,与预披露一致)

corrected max 0.909 / mean 0.363(parent mean 0.378)。按族 corrected mean:
uniform 0.105 / layered 0.349 / marmousi 0.541。0.05 绝对精度主张在本 parent
质量下不成立,如实落账;该主张的瓶颈是 parent(校正只贡献 −0.015 绝对改善),
与本线"校正头"定位一致,不构成对校正组件的否定。

## R2 校正迁移(v18 门移植):3/4 过,worst_harm FAIL

| 门 | 值 | 判 |
|---|---|---|
| joint ≥0.01 | **+0.0370**(校准面板 +0.0363,完整迁移) | PASS |
| 三族 ≥0.005 | layered +0.0592 / marmousi +0.0137 / uniform +0.0165 | PASS |
| nonworse@1% ≥90% | 449/480 = 93.5%(410/480 严格为正) | PASS |
| worst ≥ −0.02 | **−0.1183** | **FAIL** |

**尾部结构(核心发现)**:破 −0.02 底线 22 条,**全部 layered**,且集中于
parent 很好(rel_l2 0.10-0.22)的记录——与 r4e11"校正对高质量 parent 有害"
同构,hinge 在 train 侧把该尾压到 1/24,但对 validation 的好-parent layered
记录不迁移。

## 结论(该线的 validation 记录,永久)

1. 平均增益完整迁移到 unseen split(joint +0.0370,三族全过,93.5% 非恶化)。
2. 绝对 0.05 验收不成立(parent 瓶颈)。
3. 尾部风险未解:好-parent layered 记录存在最深 −0.118 的恶��,
   R2 整体判 FAIL(按冻结合取)。
4. 诚实边界:论文可主张"平均校正增益在 validation 成立",不可主张
   "无害校正";尾部需按 parent 质量门控/弃权,那是新组件,且本线
   validation 已消耗,其评估只能走 test_id 单次或新数据。
