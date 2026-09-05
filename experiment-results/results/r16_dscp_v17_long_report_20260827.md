# v17 长训报告:预算假设证伪(2026-08-27)

prereg: results/r16_dscp_v17_preregistration_20260827.json(sha fac4531a,启动前冻结)
terminal: results/r16_dscp_v17/long_ddp4/terminal.json(fail_gate / early_stop_patience,
28 epoch,wall 1215s,peak 2.48GiB,parent sha 前后一致,sealed 未开)

## 结果

- joint **+0.02028**(过门 0.01)
- per-family 三族全过:layered +0.0420 / uniform +0.0109 / marmousi +0.0080
- worst_harm −0.01234 ≥ −0.02 过
- **nonworse_within_tolerance 23/24 败**(门 24/24@容差1%):唯一违例
  train_marmousi_00365(parent 0.5824→0.5896,gain −0.0123;v16 时 −0.0058)
- 血统:epoch 1-10 monitor 与 v16 逐位一致;best@16 monitor 0.378035
  (v16 best@6 0.378386,改善 3.5e-4 = 噪声级)

## 主判:预算假设按冻结失败信号证伪

冻结判据:final joint < v16 实测 +0.0203 → "预算不是瓶颈"。实测 +0.02028 <
+0.02032,且 28 epoch 的 joint 轨迹全程在 0.014-0.021 区间震荡无趋势
(epoch 7 即达 +0.0211 峰,此后不再超越)。**2.8 倍 epoch、patience 4→12
未产生任何超出噪声的改善:wide128+192 记录配方在 ~epoch 6-7 即饱和,
长训不是该配方的瓶颈。**(注:此结论限于本配方与本面板;不与"最终精度
须经长训"的一般原则矛盾——该原则在更大数据/更大模型时才有兑现空间。)

## 证据链上的下一杠杆(按背书强度排序,均属新预注册)

1. **数据规模**:r4e13 单调曲线(24→96→192)是唯一未兑现完的正向证据;
   中间步 = fit 面板扩 ~576-768 条 + bundle fp16(~77-103GB 驻 /dev/shm,
   建缓存 ~1-1.5h),不必解决磁盘问题。
2. **C 臂 hinge**(r4e14 事后观察):joint 全 6 快照 ≥ 对照、uniform 翻正,
   单��子未证,可与 1 合并为一次预注册(hinge 是训练项改动,零额外成本)。
3. 尾部记录归因:00365 违例在 v16→v17 恶化 2 倍,与长训相关的个别记录
   过拟合迹象;数据扩张(1)天然稀释该风险。

## 记账

v17 记 1 次 fail_gate(诚实落账,阈值零移动);v16 记账不变;
产物 results/r16_dscp_v17/(terminal/best.pt/last.pt),冻结链未触碰。
