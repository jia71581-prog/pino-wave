# 论文修订记录 — 合入四组对比证据（2026-09-23）

本次把 `results/paper_comparisons_20260923/` 的四组对比产物（快照、接收器波形、粗网格频散、计时/DeepONet 表）合入 `main.tex` 并编译。定位保持"可审计诊断+范围化负证据"，不宣称精度目标达成，无端到端 10× 声称。仅修改 `main.tex`；未动实验代码、配置、检查点、原产物。

## 备份

- `revisions/20260923_before_comparisons/paper__main.tex`（SHA-256 `6024379275…`，与改前一致）
- `revisions/20260923_before_comparisons/paper__main.pdf`（SHA-256 `6313c7372a…`）
- 哈希清单：`revisions/20260923_before_comparisons/SHA256SUMS`

## 编译结果

- `latexmk -pdf`：0 error，0 LaTeX/Package warning，0 Overfull。
- Underfull hbox 6 条，与改前基线相同（同为旧表格/参考文献段落，行号平移）。合入过程中新图注一度引入 2 条新 Underfull，已通过改写图注首行消除。
- 页数 14 → 19（+5 页：新章节 §4 约 2 页文字+2 表，4 张整幅图）。

## 改动章节清单

1. **摘要**：加两句限定性描述——(i) train-development 记录上 51×51 粗解呈经典频散签名（相位滞后、谱下移）而算子晚期误差是同相幅度亏缺，且该粗解在 layered 记录上仍比算子更准；(ii) 等预算 DeepONet 基线训练塌缩，无同口径基线数字。
2. **§2.4 Runtime**（原有小节）：在 2.5×/5.1×/3.9×/7.9× 比值后加一句：这些是计时比而非精度匹配加速比——最便宜经典配置（native 网格 10.00 s）在表 2 全部三个族中位记录上都比算子更准（0.0177/0.0156/0.1197 对 0.0515/0.2135/0.4009），全文所有速度陈述均带此限定。
3. **新增 §4 "Comparison with classical solvers and an operator baseline"**（插在 §3 error floor 之后）：
   - §4 开头：声明本节全部波场图与逐记录数字来自 train-development 监测记录（9 记录 dev 集族中位），非留出集认证；被评估网络是后续 lineage `scno_homogeneous_longrun_gap15_v1`（update 5000，39,140,293 参），非 §3 锚；两图记录 future relL2 0.2785/0.1960，已按存档评估复算核对至 1e-6。
   - §4.1 粗网格频散对比：51×51 替代 50×50 的原因（求解器强制奇数节点网格）；表 5 两族全报；频散论证只由 marmousi 支撑的声明；粗解误差=时间步频散+介质采样误差的口径声明；图 3（频散谱证据）。
   - §4.2 快照与波形：图 4（marmousi 3×3）、图 5（layered 3×3）、图 6（接收器波形含 51×51 粗解叠加）。
   - §4.3 Runtime in context：仅引用 §2.4 冻结数字（无新计时），合并 record-matched 比值 5.06×/7.90×/2.52×/3.93×，两条限定（最便宜经典解更准；训练成本摊销不计，无端到端声称）。
   - §4.4 DeepONet 基线：表 6 + 塌缩事实全文如实（SIGTERM 4862/22814；12 记录 validation dense relL2 1.000012≈trivial；对照臂从未 <1.0；单记录可拟合 0.025 故裁定归优化非容量）；明确"不支持且不作'我方优于收敛 DeepONet'的排名声称"。
4. **§10 结论**：加一句限定性归纳（粗解频散 vs 神经同相幅度亏缺的性格对比；粗解在 layered 上仍胜；DeepONet 塌缩未提供收敛对比点），并补"最便宜经典配置在每个族中位上都更准"。
5. **导言区**：`\graphicspath` 增加 `../figures/comparisons_20260923/`；日期 2026-09-22 → 2026-09-23。

## 合入图表清单

| 论文编号 | 文件（复制至 `figures/comparisons_20260923/`，原产物未动） | 来源 |
|---|---|---|
| 图 3 | `dispersion_train_marmousi_00076.pdf` | group3 |
| 图 4 | `snapshots_train_marmousi_00076.pdf` | group1 |
| 图 5 | `snapshots_train_layered_00299.pdf` | group1 |
| 图 6 | `receivers_dispersion_train_marmousi_00076.pdf` | group2 |
| 表 5 | 粗网格 LWC vs 神经（两族、含时间带） | group3 NUMBERS.json |
| 表 6 | DeepONet 40M 基线（口径逐行标注） | group4 NUMBERS.json |

版面取舍：接收器波形合入含粗解叠加的 marmousi 版本一张（layered 版本未合入——其粗解无频散签名且正文/表 5 已完整报告 layered 数字）；频散谱证据合入 marmousi 一张。快照 3×3 两张全合入。`[z,x]` 轴序与既有图一致（图注声明"displayed [z,x] as stored"）。

## 数字来源（逐条）

| 正文数字 | 来源文件 |
|---|---|
| 0.2785 / 0.1960（两记录 future relL2，1e-6 复算核对） | group1 METHOD.md + group3 NUMBERS.json；原始 EVALUATION.json（staging scno gap15 update 5000） |
| 表 5 全部：0.279(0.124/0.245/0.409)、0.514(0.470/0.514/0.558)、0.120(0.122/0.118/0.119)、0.196(0.108/0.311/0.552)、0.103(0.066/0.138/0.328)、0.030(0.020/0.046/0.065) | group3 NUMBERS.json（future_relative_l2、time_bands） |
| 谱心 12.97→10.66 Hz、神经 13.07 Hz | group3 NUMBERS.json（receiver_spectrum_centroids_hz） |
| 粗解滞后 +2.5..+7.5 ms（四台三台）、神经多数台 0 ms、幅比 0.60（x=400 m） | group2 NUMBERS.json（lag_ms、amplitude_ratio）；"多数台 0 ms"沿用 group2/3 METHOD.md 口径 |
| 逐帧 relL2 0.094/0.242/0.580、0.045/0.318/1.010 及时刻 0.2175/0.575/1.0 s | group1 NUMBERS.json（frame_relative_l2、time_s） |
| 5.06×/7.90×/2.52×/3.93×、22.06/20.07/10.00/3.97/2.54 s | group4 NUMBERS.json（speed_ratios_record_matched、runtime_rows；源自冻结 runtime_20260922） |
| DeepONet 全部：39,975,985；4862/22814；1.000012（step 4488）；对照臂 1.0015–1.1168；反塌缩臂 0.97–1.33；单记录 0.025；我方 census 0.352(0.131/0.302/0.565) | group4 NUMBERS.json（deeponet 节） |
| 39,140,293 参、update 5000 | `/root/autodl-tmp/staging/scno_homogeneous_longrun_gap15_v1/run_identity.json`（既有冻结产物） |
| ~3.3 / ~4.5 ppw、vmin 1500/2068 m/s | group3 METHOD.md |

## 五条红线落实位置

- **(a) 51×51 替代 50×50 及原因**：§4.1 第一段（"The solver requires odd node-centred grids and raises on even nx/nz… nearest admissible configuration to a 50×50 grid is 51×51 at Δx=40 m"）；表 5 标题亦标注 Δx。
- **(b) layered 粗解 0.103 优于神经 0.196，频散论证只由 marmousi 支撑，两族都报**：表 5 两族整行并列（layered 粗解 0.103 加粗）；§4.1 第三段专段声明（"This dispersion argument is supported only by the Marmousi record… We report both families; the neural operator does not dominate coarse classical solves"）；摘要与 §10 结论各一句同口径限定。
- **(c) 最便宜经典解（native 201，10 s）更准，速度对比带脚注**：§2.4 runtime 段新增限定句（"These are timing ratios, not accuracy-matched speedups…"，引用表 2 六个精度数字）；§4.3 第一条 caveat 重申；表 2 本身载有该行精度。
- **(d) DeepONet 无同口径数字、塌缩如实呈现、禁排名句式**：§4.4 全段 + 表 6（标题声明"No same-caliber comparison exists… not an accuracy ranking between converged models"；行内口径逐行标注）；结尾明确"they do not support a claim that our operator outperforms a converged DeepONet, and we make no such claim"；摘要句只写"collapsed during training, so no same-caliber baseline number exists"。
- **(e) train-dev 记录非留出集认证、图注声明**：§4 开头总声明（"train-development monitoring records… not held-out certification"）+ 图 3/4/5/6 四条图注逐一写明"train-development record; not held-out certification"。

## 未改动

- 既有两图（fig 1/2）、全部旧表、§5–§9 正文数字未动。
- `README.md`、`REVISION_NOTES_20260922.md`、四组原产物目录只读未改。
- 无 GPU 作业、无 validation/test 未来波场读取（本轮纯 LaTeX/CPU 工作）。
