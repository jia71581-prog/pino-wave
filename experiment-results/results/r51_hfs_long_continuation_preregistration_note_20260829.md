# R51 HFS 长训续跑预注册说明(2026-08-29,启动前冻结)

## 假设
r39 HFS 尾部微调把 dev candidate_max 从 0.1795(r39 底座 parent)压到 0.0658,仅训练 2000 步(4x500)且 best 未见饱和证据。R46-R50 证明冻结 parent 之上的事后校正/数据加密无法闭合 0.05 max 门;唯一未测路线 = 继续微调 parent 本体。R51 用 20 倍步数(40x1000)检验 HFS 微调是否已饱和。

## 配置(与 smoke r51_hfs_smoke_1x20_v1 完全一致,仅步数)
- script: scripts/train_r39_hfs_tail_finetune.py sha256=78b4e2729336d04539907ea0b60a6d911eab67304bb7003e1c1369ace98e777a
- init: results/r39_hfs_tail_finetune_base1e6_hfs1e4_4x500_v3_20260828/best.pt sha256=98c360f1739ef86c37652166a1ee447dd85dcd260494b93601fa5b2bf6f934e2
- cache: results/r38_full_coverage_cache_20260828 (fit/holdout shards 0-3)
- 4xDDP, epochs=40, max_steps_per_epoch=1000, bs16, lr 1e-6 (HFS x100), wd 1e-4, hinge 3.0, grad 0.05, late 0.50s, seed 510828, amp
- launcher: scripts/run_r51_hfs_long_continuation.sh(启动前未改动)

## 验收 / 失败信号 / 预算 / 回滚
- 验收(与脚本内建门一致):dev holdout mean<=0.05 且 max<=0.05 双过。
- 部分成功:max 显著低于 0.0658 但未达 0.05 —— 记录为"HFS 未饱和但不足",并与 R50 弃权规则组合重新计算弃权率。
- 失败信号:40 epoch 内 candidate_max 无下降趋势(持平 0.0658)= HFS 路线饱和定论;或任一 epoch candidate_mean 恶化 >0.005 = 尾部过拟合,立即以 best.pt 为准终止分析。
- 预算:40k 步,预计 3.5-4.5 h(r39v3 实测 2000 步 614s 外推),单次,不追加。
- 回滚:r39v3 best.pt 只读未动;R51 输出独立目录;sealed(R29B/validation/test_id)不开,脚本断言 validation_opened=false。

## 数据边界
fit=训练,dev holdout=已开发集(R28 已开),评估仅此。与 R50 弃权路线并行不冲突:R51 若部分成功,弃权规则须在新 parent 上重新拟合(R50 特征依赖 parent 谱统计)。

## 启动后附注(2026-08-29 11:20,非科学修订)
实测 epoch1 = 1461s(1000 步 + 评估,~1.4s/step),40 epoch 外推 **~16h**,启动前 3.5-4.5h 的 wall-clock 估计错误(r39v3 614s/2000 步的外推不成立,本次数据加载主导)。步数预算 40x1000 与全部验收/失败信号不变,仅更正时长预期。epoch1:mean 0.02004(初始 0.02005,无恶化)、max 0.06556(初始 0.06582),暂无趋势信号。

## 终止记录(2026-08-29,lead 指令截断)
lead 指令"调整学习率和网络结构",于 epoch 5(step ~4600)截断。截断时证据:5 epoch dev mean/max 双平(mean 0.01996-0.02008,max 0.06534-0.06607 振荡带),训练 loss OLS 斜率与半段差均在噪声内,难度归一 loss 彻底走平。结论(降级为部分证据,非满预算定论):**lr 1e-6 下 HFS 续训无信号**,与失败信号 A 形态一致但未跑满 40 epoch。best.pt(ep4,score 0.0853)保留于输出目录。后继实验 = R52(lr 阶梯 + 空间化 HFS 结构)。
