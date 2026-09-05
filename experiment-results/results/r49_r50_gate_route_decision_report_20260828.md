# R49/R50:0.05 开发门两条路线的判定(2026-08-28)

## 背景
- R46(21.2M 残差 FNO)、R47(201K 块复增益 refiner)、R48(fit-only 全局/族增益校准)三次独立尝试同一失败模式:oracle 容量存在(逐记录块增益可达 dev max 0.045-0.047),但一切从 fit 学到的记录无关映射迁移为零(dev max 恒 ~0.0658,r47 holdout 自选 scale=0)。
- 剩余两条路线:①改 fit 分布(数据加密);②可部署难度检测 + 按记录弃权。R49/R50 分别用纯分析探针(零训练、零 GPU)判定。

## R49:最近源组增益空间迁移探针(数据路线) — 证伪
脚本 `scripts/analyze_r49_nearest_group_gain_transfer.py`,产物 `results/r49_nearest_group_gain_transfer_v1_20260828.json`。
donor 选择只用部署可得的源位置;增益只来自审计过的 fit union 真值。

- fit 侧 LOGO(逐组留一,knn1,~75 m 最近邻):mean_imp **-0.071**,84% 记录净伤害。
- dev 侧单 donor 距离阶梯(全部 marmousi):50 m 档 mean_imp -0.028(69% 伤害),各距离档全负。
- 最难组 x6800:z100 的 8 个最近 donor(71-150 m)全部无益(最好 +0.005)。
- knn 池化最优配置(knn8_idw,16 块)dev max 0.0653 vs parent 0.0658,gate FAIL。

**结论:块增益场在 <50 m 源间距尺度上已去相关。现有 fit 网格 50-70 m,任何可行密度的数据加密都无法用邻组增益闭合 0.05 max 门。数据路线死。**
(佐证:fit 难度与源位置 x 的 Spearman 仅 0.34,fit 自身 p99=0.047、max 0.066,>0.05 有 7 条——即使 fit 分布内该门也不整体成立。)

## R50:可部署难度检测 + 弃权探针 — 路线成立
脚本 `scripts/analyze_r50_deployable_abstention_probe.py`,产物 `results/r50_deployable_abstention_probe_v1_20260828.json`。
特征只用部署可得量(parent 谱统计 13 维:加权能量、DCT 频带占比、频率质心、静态介质通道高频占比、源 f0/t0);真值只作 fit 标签与 dev 诊断。ridge(λ=1)拟合 log parent rel L2。

- 检测器排序质量:fit Spearman 0.851,**dev Spearman 0.897(迁移完好,甚至更高)**。
- dev 侧权衡曲线:弃权 top-6 记录(10.7%)即 kept_max **0.0448**,2 条 >0.05 记录排名 3/6 全部命中。
- fit-only 冻结规则(部署可用,无 dev 调参):
  - conformal q99(fit 残差 99 分位余量 0.883,弃权线 pred*margin>=0.05):dev 弃权 19.6%,kept_max **0.0448**,miss 0 → **门过**。
  - 逐族检测器 + lab 0.045:marmousi dev 弃权 10%(miss 1 条 0.0511 压线),layered 弃权 55% miss 0 → 全局 conformal 优于逐族。
- 诚实风险:fit 侧存在 2 条 layered 检测盲点(rel 0.051/0.066,检测分位仅 0.57/0.22),规则在 fit 自身 miss 2;dev 上未显形但不能声称零漏检,只能声称"q99 conformal 在 dev 上实现 kept 全体 ≤0.05、弃权 19.6%"。

## 判定
1. 0.05 无条件 max 门在本 parent(r39 底座)下**不可达**:oracle-learned 迁移缺口(R46-R48)+ 增益场亚网格去相关(R49)双重证据。
2. 可交付定位改为**选择性预测**:R50 的 13 维可部署检测器 + fit-only conformal 弃权是首个门内方案(dev:coverage 80.4%,kept_max 0.0448)。
3. 后续若走正式确认:需预注册 R50 规则(特征、λ、q99 余量全冻结),在 final_train_confirm 或经 lead 解封的一次性面板外评估;layered 盲点需在预注册中列为已知风险。

## 数据边界
fit 真值用于标签/增益估计;opened group-disjoint development 仅诊断;R29B、final validation、test_id、论文全程未动。零 GPU、零训练。
