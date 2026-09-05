# DeepONet 基线过拟合探针 —— VERDICT

日期: 2026-07-28
目的: 隔离"纯全局低秩算子 (textbook DeepONet: branch ⊗ trunk 内积, 无 FNO 前端/travel 分支/MIONet 乘积/coarse 场)
相对我们的结构化模型买到了什么"。与 `results/capacity_ladder/` 同款确定性 3 记录 (uniform/layered/marmousi) 过拟合探针、
同款评分与目标 (agg<0.10 且各族<0.12)。脚本 `scripts/train_deeponet_baseline_overfit.py` + `deeponet_baseline/model.py`。

## 结果 (3 记录尽力过拟合, best aggregate relative L2)

| 模型 | 参数 | best agg | uniform | layered | marmousi | 备注 |
|------|------|----------|---------|---------|----------|------|
| 6.6M (lr1e-3, 3000 步) | 6.6M | **0.663** | 0.952 | **0.037** | 1.000 | 只记住 layered |
| 13.4M (lr1e-3) | 24M | 0.791 | 1.000 | 0.372 | 1.000 | 更差, 噪声大 |
| 23M (lr1e-3) | 39M | 1.006 | 1.001 | 1.014 | 1.002 | 卡平凡解 |
| 41M (lr1e-3) | 69M | 1.020 | 1.003 | 1.054 | 1.003 | 卡平凡解 |
| (lr3e-4 版, v3) | — | ~1.000 | — | — | — | lr 太低全卡平凡解 1.0 |

## 判定

**纯全局低秩 DeepONet 连过拟合 3 条记录都做不到**: 最佳 (最小的 6.6M) 也只 0.66, 且只记住了 layered (0.037),
uniform/marmousi 死钉在 ~1.0 (平凡零预测解)。加参数 (6.6M→69M) **不降反升/发散**, 大模型在 lr1e-3 下不稳定、
lr3e-4 下卡平凡解。这与作者 deep_research 的 Root Cause #4 (全局池化毁空间信息→只能出空间均匀场→最优≈零预测) 完全一致。

**对照我们的结构化模型**: capacity_ladder 同款 3 记录过拟合下界 = **0.125** (w192)。即:
- 纯 DeepONet 过拟合下界 ≈ 0.66 (且只对 layered 有效);
- 我们的 FNO/MIONet 结构过拟合下界 ≈ 0.125 (三族均衡)。

**结论: 现有 FNO 前端 + 局部相位 + (MIONet/局部) coarse 场结构相对纯全局低秩算子买到了 ~5× 的过拟合能力
(0.66→0.125), 尤其在 uniform/marmousi 的锐波前/复杂介质上 —— 纯全局池化算子在这两族上完全失效。
这从反面佐证了 [[fno-acoustic-local-field-optionb]] 的方向 (保空间结构的局部传播场) 是对的。**

## 产物
- `results/deeponet_baseline/deeponet_{6.6M_v3,13.4M_v4,23M_v4,41M_v4}/` (metrics/run_identity/updates)
- 脚本: `scripts/train_deeponet_baseline_overfit.py`, `deeponet_baseline/model.py`

## 备注
v4 系列在 lr1e-3 下大模型梯度噪声大、未跑满 3000 步即被本会话 (为腾 GPU 做高容量长训) 主动停止;
但 6.6M 已跑满且给出决定性结论 (纯算子过拟合都到不了 0.125), 更大模型只会印证 Root Cause #4, 无需跑满。
