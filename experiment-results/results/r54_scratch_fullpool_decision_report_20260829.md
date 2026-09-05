# R54 从零全池重训 终判 (judged 2026-08-29 23:50 CST)

预注册: results/r54_scratch_fullpool_preregistration_20260829.json (frozen_before_training, amendment_v2 infra-only)
运行: results/r54_scratch_fullpool_40ep_v2_20260829 (exit 0, 40/40 epochs, 9307s, 4xGPU DDP)

## 判定
- failure_signal (max < 0.06894): 通过 — best candidate_max 0.06478 @ ep15
- signal_gate (max < 0.06582, r39 链最优): **通过 — R54 成为新链最优交付件**
  新 chain-best checkpoint: r54_scratch_fullpool_40ep_v2_20260829/best.pt
  (sha256 2a544eac...; 取代 r39 v3 0.06582)
- success_gate (mean<=0.05 AND max<=0.05): mean 0.01775 通过 / max 0.06478 未过 → **FAIL**
  → 按 evidence_boundary, R29B fresh-holdout 不解封, validation/test_id 保持封存

## 饱和判读
best @ ep15; ep16-40 共 25 评估轮无 >5e-4 改善 (holdout 早饱和)。
train_loss ep40=0.000717 仍缓降 (末 8 轮降幅 ~3%),但 holdout 平坦 → 记 holdout_saturated_at_ep15,
不构成 still_descending 延展理由 (延展只会尾部过拟合, 见预注册 r28_trajectory_caveat)。

## 数据杠杆定论
fit 896→2128 (2.38x) + ep 12→40: max 0.06895→0.06478 (相对 -6%)。
预训练侧数据量是弱杠杆; per-family max: uniform 0.0126 / layered 0.0327 / marmousi 0.0648。
全局 max 完全由 marmousi 尾决定 — 与 R50/R52/R53 结论一致: 剩余瓶颈非容量非数据量非微调,
是 marmousi 族 parent (LWC-84 coarse) 误差水平 (parent_max 0.1795)。

## R55 冻结规则套 R54 (refit_performed=false)
r54_scratch_fullpool_40ep_v2_20260829/r55_selective_prediction.json:
- conformal: 弃权 71.4% (16/56 kept), kept_max 0.00618, 零漏检 (0.05 门)
- threshold: 弃权 80.4%, kept_max 0.00571, 零漏检; marmousi 100% 弃权
当前最强可交付 = R54 best.pt + R55 conformal 弃权规则。

## 链状态
微调树 (r39 系) 与预训练侧 (R54) 两侧杠杆均已拉尽且饱和。
0.05 无弃权全覆盖目标在现架构+现 parent 下不可达; 达标路径只剩:
(a) 换/强化 marmousi parent 数值解 (提高 LWC-84 精度或分族 parent), 或
(b) 接受选择性预测交付 (R54+conformal)。
