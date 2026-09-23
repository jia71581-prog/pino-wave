# 审稿资料包（更新至 2026-09-23）

本目录保留当前论文所用的代码快照、冻结协议、最终诊断数值、计时对照与图件。完整研究说明在仓库根目录的 [RESEARCH_OVERVIEW.md](../RESEARCH_OVERVIEW.md)，论文见 [main.pdf](paper/main.pdf)。

| 路径 | 内容 |
|---|---|
| `code/` | grouped UFNO / MIONet 模型、数据接口、训练入口、匹配配置及 33 文件来源哈希 |
| `protocols/` | 冻结基线、排除组和预注册；`GATE_RESULT.json` 是最终门限裁决 |
| `results/validation_480_summary.json` | A4 step 22814 的 480 条 validation 记录摘要 |
| `results/diagnostic_audit_20260922/` | 29359 锚模型的时间误差算术、部署空间校正方向诊断及来源 |
| `results/runtime_20260922/` | 冻结计时结果和基准脚本 |
| `results/paper_comparisons_20260923/` | 四组最终图表、数值、方法和生成脚本 |
| `paper/`、`figures/` | 当前论文、修订记录及编译所需图件 |

未收录中间失败试验的过程文件，也未收录原始 `.npy` 数组、数据集或权重。组 4 的 DeepONet 表格记录的是**最终对照结论**：该基线未收敛，不能用于已收敛模型的性能排名。论文中的 train-dev 图表不是留出集认证。比较脚本带有原机器绝对路径，异地复跑需提供对应数据和检查点并调整路径。

在此目录运行 `sha256sum -c SHA256SUMS` 与 `python3 verify_package.py`。哈希清单覆盖包内所有发布文件，但不包含清单自身。`verify_package.py` 只复核代码来源、频带恒等式和条件反事实，不会运行模型。
