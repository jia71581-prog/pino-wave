# PINO-Wave：二维声波神经算子研究

本仓库的最新审稿资料是 [REVIEWER_PACKAGE_20260921](REVIEWER_PACKAGE_20260921/)；资料于 2026-09-21 建包，论文和对照证据更新至 **2026-09-23**。

- [完整中文研究说明](RESEARCH_OVERVIEW.md)：任务、数据、模型、验证口径、结果、局限与复核方法。
- [论文 PDF](REVIEWER_PACKAGE_20260921/paper/main.pdf) 与 [LaTeX 源文件](REVIEWER_PACKAGE_20260921/paper/main.tex)：当前 19 页稿件。
- [产生报告结果的代码快照](REVIEWER_PACKAGE_20260921/code/) 与 [来源哈希](REVIEWER_PACKAGE_20260921/code/PROVENANCE.tsv)。
- [最终证据与对照](REVIEWER_PACKAGE_20260921/results/)；[冻结协议](REVIEWER_PACKAGE_20260921/protocols/)；[图件](REVIEWER_PACKAGE_20260921/figures/)。
- [原项目源代码](grouped-dual-head-v2/) 保留，包含更广的历史实现和数值数据生成程序。

**当前状态：目标未达成。** A4 step 22814 的 480 条 validation 记录，逐记录等权未来场相对 L2 均值为 **0.3522**，uniform/layered/Marmousi 分别为 **0.1309/0.3020/0.5654**。项目的 5% 精度和端到端 10× 加速联合门限未通过；`test_id` 未用于本轮结论。训练面板的诊断数字与这组 validation 数字属于不同模型谱系，不应合并。

这次公开版只收录最终论文、代码快照、冻结协议、可核验的诊断和对照产物。旧的 `experiment-results/` 历史过程归档已从当前分支移除；需要历史记录时可查看先前提交。中间失败的运行目录、临时日志、原始 `.npy` 波场、HDF5 数据集及训练权重均未纳入当前版。最终的负面研究结论仍在论文和研究说明中如实报告。

## 快速核验

```bash
cd REVIEWER_PACKAGE_20260921
sha256sum -c SHA256SUMS
python3 verify_package.py
python3 results/diagnostic_audit_20260922/recompute_late_floor.py
```

前两项核对发布文件和 33 个代码文件的来源，第三项复算已发布标量的晚期误差反事实。重新运行神经网络预测需要另行取得数据、检查点及相同的运行环境；单靠本仓库不能重训或重现整场预测。
