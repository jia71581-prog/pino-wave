# Transfer DG 固定 skeleton 真实通量 oracle 结论

## 判定

**拒绝当前 56-DOF、200 m 双线性 skeleton pressure lifting。**

该实验只在三组 train-only group-disjoint holdout 上运行。未来真值用于构造 exact flux residual，因此结果只能用于容量判断，不能部署或晋升。`validation_opened=false`，`test_id_opened=false`。

## 结果

| 指标 | 父模型 | 固定 skeleton oracle |
|---|---:|---:|
| 72 窗口 aggregate relative L2 | 0.28002795 | 1.48806473 |
| 0.60/0.25/0.15 weighted score | 0.20557475 | 1.07174362 |
| slot 0 | 0.11324500 | 0.54948847 |
| slot 1 | 0.28601923 | 1.54844677 |
| slot 2 | 0.44081962 | 2.36625894 |
| low frequency | 0.26729147 | 1.35847520 |
| middle frequency | 0.39023024 | 2.37749995 |
| high frequency | 0.55722957 | 3.07016660 |

- 改善记录：0/72。
- 变差记录：72/72。
- 平均 skeleton flux residual 降低：11.07%。
- 平均 pressure correction / parent norm：1.633。
- aggregate 相对变化：-431.4%（误差增加）。

## 解释

投影矩阵为 2540 个内部面通量约束、56 个压力自由度，满列秩，条件数 8.05；单元测试证明它可精确恢复自身双线性试验空间。因此失败不是明显的符号、秩或求解器错误。

真实波场通量残差的大部分位于当前粗试验空间之外。最小二乘为了拟合约 11% 的可表达通量分量，生成了幅值过大的平滑 pressure correction，导致低、中、高频和全部记录同时恶化。通量 residual 降低并不保证 pressure relative L2 降低。

## 决策

1. 不继续训练当前 local DtN 输出后接 56-DOF pressure lifting。
2. HCAIS 采样器保留，因为其独立配对实验已改善 local flux holdout。
3. DG 信息下一步应作为全局预训练中的监督辅助目标，直接约束父模型 pressure gradient/interface flux，而不是先预测 flux 再做粗 Neumann-to-pressure 重建。
4. 若未来重新测试 skeleton，需要 coefficient/frequency-adapted plane-wave trace basis 和能量受控的组装，并重新做 train-only oracle；不能通过 truth-selected scale 挽救当前版本。

