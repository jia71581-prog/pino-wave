# R52 预注册:lr 阶梯 + 空间化 HFS 结构(2026-08-29,启动前冻结)

## 动机
R51 证明 base lr 1e-6 下 HFS 续训 5 epoch 无信号(mean/max/loss 三平)。两个从未测过的调整:
- **A 臂 lr**:r39 全部版本 base lr 均为 1e-6,×10 从未试过。HFS 参数 lr 固定 1e-4 不变(multiplier 100→10)。
- **B 臂结构**:HFS 从逐通道标量(1,120 参)升级为逐通道 8×8 空间 λ 图 + 双线性插值到 patch 网格(71,680 参)。identity 语义保持;从 r39v3 热启动 = 标量广播进图,已 CPU 核验函数逐位一致(max diff 1.5e-8)。

## 配置
- 共同:script 基座 train_r39_hfs_tail_finetune.py(sha 78b4e272)不改;init = r39v3 best.pt(sha 98c360f1);cache = r38_full_coverage_cache;4×DDP,4 epoch × 1000 步,bs16,wd 1e-4,hinge 3.0,grad 0.05,late 0.50s,amp。
- A 臂:scripts/train_r52_hfs_lr10.py,base lr 1e-5,HFS multiplier 10(HFS lr 恒 1e-4),seed 520829。
- B 臂:scripts/train_r52_spatial_hfs.py,SPATIAL_GRID=8,HFS multiplier 100,seed 520830。base lr 规则(冻结):A 臂若无任一 epoch mean ≥ 0.0250 → B 用 1e-5,否则 1e-6。
- B 臂长训前先 1×20 smoke,门:initial_holdout candidate_max 与 0.065821 差 ≤ 1e-4(函数保持核验),不过则 B 臂废止。

## 判读(冻结)
- 改善信号:任一 epoch dev candidate_max ≤ 0.0630(清出 R51 振荡带 0.0653-0.0661 两倍幅度)。
- 过冲失败:任一 epoch candidate_mean ≥ 0.0250(初始 0.02005 + 恶化裕度)。
- 双臂均无改善信号 → 定论"r39v3 warm-start 微调路线饱和(lr 与 HFS 容量双证)",可交付回 R50 弃权方案。
- 任一臂出改善信号 → 该配置进入新预注册的长训,长训过 0.05 门才算数;本 4-epoch 结果不作达标声明。

## 预算 / 回滚 / 边界
- 预算:A 1.6h + smoke 4min + B 1.6h ≈ 3.3h,单次不追加。
- 回滚:r39v3 best.pt 只读;各臂独立输出目录;R51 产物保留。
- 数据边界:fit=训练,dev holdout(R28 已开)=评估;R29B/validation/test_id 不开。
