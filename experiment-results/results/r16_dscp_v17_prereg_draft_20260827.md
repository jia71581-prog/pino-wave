# r16_dscp v17 预注册草案(2026-08-27,待 r4e14 判读后冻结)

## 候选与定位

- candidate: `r16_dscp_v17_wide128_longtrain`
- 性质:输出场数据拟合 + 特征驱动 ansatz 校正,无 PDE 残差(v16 免责声明逐字继承)。
- 阶段声明:train 侧规模化泛化;0.05 验收主张仍需未来一次性 sealed validation。
- v16 记账不变(fail_gate);v17 为新预注册,阈值调整全部在此披露,不回改 v16。

## lead 指令依据(2026-08-27)

1. "我们的最终精度是长训的结果,而非几步就到" → v16 long patience=4 在 epoch 10
   早停(用掉 447s/7200s)不足以体现配方精度;v17 大幅加大训练预算与耐心。
2. "不用花时间解决内存不够的问题" → 保留 long_ddp4 的 rank%3 shard 分片方案原样,
   不做全 shard 加载改造。

## 结构(相对 v16 long_ddp4 的全部差异)

1. **训练预算(主变更)**:max_epochs 20→100,patience 4→12,wall_s 7200→14400。
   updates_per_epoch 1536、lr 3e-3 恒定(r4e12:cosine 无收益)、AdamW/clip/seed 372、
   冷启动、Wide128Head(25266 参)全部不变。
2. **no-harm hinge(槽位,依 r4e14 判读)**:
   - E5a → 训练损失 = masked_confined_loss.total + λ·relu(cand_rel/par_rel −(1−m)),
     λ、m 取 r4e14 获胜臂配置,逐字披露;
   - E5b/E5c → 不加 hinge,v17 与 v16 唯预算与门定义不同。
   [待填:r4e14 判读 + 获胜臂]
3. **nonworse 门重定义(披露的容差重议)**:v16 零容差 cand≤par 计数 ≥23/24 改为
   **gain_iii ≥ −0.01(1% 相对容差)且 24/24 全数满足**。
   依据与披露:v16 long_ddp4 的 9 条 worse 全部 gain_iii ≥ −0.0059(先验上,校正
   对 0.05-0.17 量级 parent 的 <1% 相对扰动 = 绝对 0.0005-0.0017,相对 0.10 目标
   不构成实质危害);本定义在 v17 任何一次评分前冻结,属"新预注册 + 披露先验来源",
   不是把 v16 门向 v16 实测值移动(v16 结论不追改)。同时新增次级门
   **worst_harm ≥ −0.02**(单记录最大相对恶化 2% 封顶)防尾部。
4. **跳过 pilot,链 = smoke → long_ddp4**:r4e13 已证 24-fit pilot 数据饥饿且误导
   (A24 −0.033 vs C192 +0.023 单调),它不再是 long 的有效前置证据;smoke
   (3 记录 wiring + oracle fraction 门,v16 逐字)保留作接线检查。

## 门(long_ddp4,全部在冻结时定死)

- joint_improvement ≥ 0.01(承 v16)
- per_family_improvement ≥ 0.005 三族(承 v16)
- nonworse:gain_iii ≥ −0.01 达 24/24,且 worst_harm ≥ −0.02(新,上节披露)
- finite / VRAM ≤ 8 GiB / checkpoint ≤ 2 MiB(承 v16)
- 数据角色:fit=long_fit(192),confirm=long_calibration(24),truth 只开 train
  侧 allowlist;validation/test_id 保持 sealed。
- monitor:confirm 平均校准损失(承 v16),best.pt 按 monitor 选择;patience 12。

## 失败信号与回滚

- 失败信号:100 epoch 或 patience 触发后任一门未过 → fail_gate 诚实落账;
  若 joint < v16 的 +0.0203 → 长训假设证伪,写入"预算不是瓶颈"。
- kill-switch:训练发散(非有限损失/梯度即刻 Refusal)、VRAM 超门、磁盘 <2 GiB。
- 回滚:v17 产物独立目录 results/r16_dscp_v17/,不触碰 v16 冻结链与 parent
  (448035bd,运行前后 sha 必须一致)。

## 预算

单链一次:smoke ~130s + long_ddp4 ≤4h wall(预计 ~90-110 min 实际:153.6k upd
@~35 步/s + 100 次分布式评估)。GPU 4×4090D,VRAM ~2.5 GiB/卡。磁盘增量 <10 MB。
