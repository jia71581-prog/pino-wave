# r4e14 报告:no-harm hinge 与长训地平线(2026-08-27)

spec: results/r4e14_noharm_hinge_spec_20260827.md(sha 5653b53e)
terminal: results/r4e14_noharm_hinge_20260827/terminal.json(status success,
3 臂 × 12288 upd,fit 144 / holdout 48,单 GPU,~17 min)

## 判读(按冻结判据,最终快照 upd 12288)

| 臂 | joint | harms/48 | worst | harm_mean | uniform pf |
|---|---|---|---|---|---|
| A 对照 λ=0 | +0.0105 | 20 | −0.134 | −0.0240 | +0.0016 |
| B hinge λ=1 | +0.0141 | 17(−15%) | −0.096 | −0.0178 | +0.0063 |
| C hinge λ=1 m=0.02 | +0.0162 | 15(−25%) | −0.097 | −0.0229 | +0.0063 |

- **主判 E5b**:B/C 危害数降幅(15%/25%)均未达 E5a 的 ≥30% 线 → 按冻结判据
  hinge 不纳入 v17。v17 已以 hinge 关闭冻结并启动,与判读一致。
- **Q2(长训)成立**:A 臂快照 4→6 joint +0.0078→+0.0105(+0.0027 ≥0.002),
  轨迹仍在爬 → v17 加大预算有据。
- **事后观察(仅作 v18 假设,不构成证据)**:C 臂 joint 在全部 6 个快照上 ≥ A
  (最终 +0.0162 vs +0.0105,+54%),且 uniform per-family 由 A 的 ~0/负翻正
  (+0.0063);B 臂 harm_mean 最低(−0.0178)。单种子、多重比较未校正,
  若 v17 预算实验后仍需结构项,v18 候选 = C 配方(λ=1, m=0.02)重测。

## 附带发现

- **held-out 危害远大于 v16 confirm 所示**:train 侧 48 条 holdout 的 worst harm
  达 −0.13~−0.28(v16 confirm 24 条仅 −0.006),且集中于个别记录;
  v16 的 confirm 面板对尾部危害的暴露不充分,v17 的 worst_harm 门(−0.02)
  在该面板上仍可能是弱约束——解释权留给 v17 实跑。
- 三臂 joint 轨迹在 upd 10240 处同现凹陷(A +0.000/B +0.008/C +0.010)后回升,
  同种子同序下属采样噪声,非 hinge 效应。

## 卫生

truth 只经预建 shard(train/long_fit),零新开;long_calibration/pilot_confirm
未触碰;v16/v17 冻结链未写入;peak VRAM 2.6 GiB;判读先验冻结于 spec,
无阈值后移。
