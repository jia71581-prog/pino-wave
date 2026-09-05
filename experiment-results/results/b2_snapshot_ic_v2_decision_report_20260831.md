# B2快照IC v2 终判 (judged 2026-08-31 00:2x CST)

预注册: results/b2_snapshot_ic_preregistration_20260830.json
(amendment_v3 = FrameConditionedPropagator结构修正,判据/预算/manifest/种子与冻结版一致)
四lane: results/b2_snapshot_ic_v2_{ic2,ic8}_s{372,733}_20260830 (60/60 ep, exit complete)

## 判定 (baseline agg 0.12603, gate A线 0.12403)
- **gate A (beats anchor): 双臂 PASS**
  ic2: s372=0.12096 / s733=0.11897 (均<0.12403), 臂均值 0.11997
  ic8: s372=0.11779 / s733=0.11679 (均<0.12403), 臂均值 0.11729
- **gate B (IC帧数差异): PASS — ic8 优于 ic2**
  |mean差| 0.00267 > max种子带 0.00199, 且两种子配对方向一致 (−0.00317 / −0.00218)

## 结论
1. anchored-B2修正杠杆成立: 相对warp_r1锚 −4.8%(ic2) / −6.9%(ic8)。
   v1欠拟合诊断得到确认: 同判据同预算下, 唯一差异是decode见base_k+3x3感受野
   (amendment_v3, 两子因素捆绑未归因)。
2. 8帧IC优于2帧IC, 按冻结判据双重成立。这超出audit_residual_memory.py的
   m<=4实测范围, 属新证据; 但该audit测的是物理核残差, 本线是纯学习传播子, 两者
   不矛盾。
3. 未饱和信号: 四lane的best全在ep60(最后一轮), cosine退火末段仍在下降,
   60ep预算截断了轨迹。per no_post_hoc_extension, 不延展本轮; 续训须新预注册。
4. 遗留瓶颈与R54链同构: max恒0.503(worst记录几乎未动), 修正是均值侧的;
   per_family: uniform 0.084 / layered 0.129 / marmousi 0.138-0.149。

## 证据边界
development 90条子集only; validation全量与test_id保持封存。
