# R6 细网格残差缓存：研究决策（2026-09-05）

六条记录的 float16 缓存 smoke 已在独立审计 CLEAR、lead 单独 release 后完成：5/6 条记录违反预注册残差信噪门，判定 `cache_representation_rejected`。这拒绝的是分别存储两个 float16 场的表示，不是算子路线。完整缓存构建、训练、confirmation、validation、test_id 或晋级均未启动。所有旧检查点及本次 smoke 产物保留，不重试、不改变门槛。

## 已有证据及其边界

- R6 是纯数值求解器，使用 5 m 内部网格、625 μs 时间步，并把结果限制到 201 × 201 输出网格。它不是已经取得高精度泛化成绩的神经算子。冻结回退定义为 `results/frozen_fine_grid_dt625us_candidate_r6_20260815.json`，SHA256 为 `a83aeeb94cb61fbb56d18562074356a75875c7a75b7d7327b64d0cd0ed9c2947`。
- 旧 R54 学习粗数值解的残差，在 R38 development 的 56 条记录上平均相对 L2 为 0.0177470、最大为 0.0647763，尚未满足逐样本 5% 门槛。这些是已用于开发的数据；不能当作独立泛化成绩，也不能转移到当前 R6 残差。来源：`results/r54_scratch_fullpool_40ep_v2_20260829/terminal.json`。
- 最新 R6 加新鲜、未训练 R54 头仅完成运行时与身份检查，约 1.287 s，相较 R6 约 0.89 s 增加约 45%。运行时可接受不证明精度改善。来源：本 candidate 的 `runtime_report.json`（SHA256 `6c221ae7b882f53aa0e333c9c191cf76d05bd57dd50b8950535b0a2fce4927b5`）。
- 数据审计发现，R38 development 与 R29B 各有 4 个 Marmousi crop；它们相对 fit crop 的最大面积重叠均为 95.0625%。group id 不重叠不足以证明地质分布外泛化。当前源幅度恒为 1，Ricker 主频约 8–30 Hz，深度 50–300 m，且 `t0 * f0 = 1.5`，源族覆盖仍有限。
- 当前 `_fine_velocity` 从 manifest 和原始 Marmousi 材料重构内部网格，不接受任意 201 × 201 速度图作为唯一充分输入。相关输入必须纳入未来部署和泛化定义。

## 本次可证伪假说

对当前 R6 微小残差分别存储两个 float16 场后，训练器可见的 `qT - qP` 仍保留足够的原生 `T - P` 信号。风险是两个场各自的量化误差相对于微小残差过大。最低成本证据只需 3 个介质族各 1 条 fit 和 1 条 development，共 6 条已冻结 train 记录，无需训练。

P/T 为原归一化 float32 场；qP/qT 必须来自实际 HDF5 float16 写入、flush 和回读，再转 float32。差分、内积及范数累计用 float64。诊断复用 parent 完整预测哈希之后的同次 train truth 读取，额外 truth 读取数为 0。报告每条记录及 early/mid/late 的 `R_native`、`R_cached`、`E_q`、`E_q / R_native`、残差余弦、qP/qT 自身场误差和 static7 回读一致性。

逐完整记录要求 `E_q < 1e-3`；当 `R_native >= 1e-5` 时，还要求 `E_q / R_native <= 0.25`。时间带只用于诊断，近零残差带不另设拒绝门。零真值必须显式验证量化后仍为零；零分母返回 JSON null，并保留原始平方范数。static7 量化回读字节必须一致。任一门槛失败仅判定 `cache_representation_rejected`，保留全部 smoke HDF5、summary 和 terminal；不判定算子研究失败。

smoke 预算为 GPU 0、180 s、峰值低于 8 GiB。现阶段保持原 float16 格式。若表示被拒绝，下次才可预注册单一变量的 float32 父场/真值缓存对照；当前不自动改变存储、模型、损失或预算。

## 数值频散与后续泛化方向

应分开研究 R6 的时间离散及微残差，与真正的粗空间网格频散。物理审计用现有 `tgrs_dclp_no/dispersion.py` 的 `phase_velocity_ratio` 对均匀介质、内部离散符号、轴向传播、c = 1500 m/s 重算相速滞后：

| 主频 | h = 5 m, dt = 125 μs | h = 5 m, dt = 625 μs | h = 10 m, dt = 125 μs | h = 10 m, dt = 250 μs |
| --- | ---: | ---: | ---: | ---: |
| 25 Hz | 0.843 ppm | 0.972 ppm | 178.999 ppm | 179.002 ppm |
| 50 Hz | 179.002 ppm | 181.073 ppm | 22112.533 ppm | 22112.578 ppm |

这些数值不包含异质介质、界面、源离散、边界或限制算子的误差，不能解释总误差，也不能把 R6 残差校正称为已经修正了粗空间频散。数值传播加 FNO 频散修正是可检验的后续假说；[原始研究](https://arxiv.org/abs/2510.06881) 的三维弹性结果不能作为本项目二维声波成绩。

未来泛化评估需要新合成速度模型、独立组合的震源参数及地质隔离；不能只使用高度重叠的 Marmousi crop 或现有反复开发的 holdout。当前 stage 不读取新评估真值、不改变论文。

## 本次修改与复审材料

旧 preregistration、dependency manifest、master reference 以及 builder/test 原文件保存在 `audit_v2_20260905/original/`，均为只读快照。修复只补齐已有 binding 中缺失的 10 条路径，严格哈希相等与递归依赖检查保持不变。代码修改限于 builder 的 smoke 量化诊断和 CPU 回归测试。精确命令、代码与协议 diff、CPU 测试报告及不可变 smoke 输入快照位于 `audit_v2_20260905/`。

冻结 preregistration 的 status `runtime_passed_cache_smoke_pending_audit` 保留为准备时快照，实际 smoke 授权单独记录于 `audit_v2_20260905/lead_smoke_release_v2.json`；不能解释为完整缓存授权或精度验证通过。

## 已完成的六条记录 smoke

GPU 0 上的运行耗时 27.414 s，PyTorch 峰值已分配显存 352,735,232 bytes（约 0.329 GiB），符合 180 s / 8 GiB 预算。运行中的 `nvidia-smi` 独立观察到 child PID 691347、约 886 MiB 进程显存。supervisor PID 691346 已回收 child，退出码为 2；两进程均已退出，四卡最终均为 0 MiB / 0%。没有训练或新检查点。

以下指标均为相对 L2 的无量纲小数，时间带详细结果见两份 role summary 和 `audit_v2_20260905/smoke_writer_postrun_v2.json`：

| role / family / sample | R_native | E_q | E_q / R_native | 残差余弦 | record gate |
| --- | ---: | ---: | ---: | ---: | --- |
| fit / uniform / train_uniform_00000 | 0.000206048 | 0.000237123 | 1.150814 | 0.656597 | 拒绝 |
| fit / layered / train_layered_00000 | 0.0000386383 | 0.000118141 | 3.057602 | 0.310691 | 拒绝 |
| fit / marmousi / train_marmousi_00000 | 0.000691098 | 0.000270657 | 0.391633 | 0.931128 | 拒绝 |
| development / uniform / train_uniform_00357 | 0.000410503 | 0.000265458 | 0.646665 | 0.839706 | 拒绝 |
| development / layered / train_layered_00548 | 0.001459115 | 0.000239273 | 0.163985 | 0.986821 | 通过 |
| development / marmousi / train_marmousi_00275 | 0.000125686 | 0.000206325 | 1.641587 | 0.520957 | 拒绝 |

全部 6 条的 `E_q < 1e-3`，但 5 条不满足 `E_q / R_native <= 0.25`，最严重的 layered fit 中量化扰动约为目标残差的 3.06 倍。全部 static7 的实际 float16 回读字节一致，零真值位置量化后仍为零；legacy/device 父场逐字节一致。truth ledger 证实仅 6 次 train 读取、1395 帧、225,437,580 原始 bytes，每条均先完成父场哈希，且只读一次。量化诊断未追加 truth 读取，所有冻结输入哈希不变。

`smoke_terminal.json` 的 SHA256 为 `4aaaa5c677654e4e4fc296de0337787be4deaf631618ea531c9f8b14a94b4094`。两份 HDF5 共 89,536,754 bytes，保留于 `smoke/`；exact argv、release、detached identity、live observation、exit record、postrun report 和日志均保留于 `audit_v2_20260905/`。旧 preregistration、依赖、代码与快照没有改动。

下一最低成本对照是在独立 stage 以相同六条记录，仅把父场/真值缓存改为 float32，static7 仍保持 float16，检验量化噪声是否消失。必须另行冻结输入、预注册并复审后才能运行；本次没有启动这个后续对照，也没有据六条样本宣称达到泛化或数值频散目标。

---

**后续结果链接（2026-09-05 追加，不改动上文）**：上述"下一最低成本对照"已在独立 stage 完成。候选 `r6_device_residual_cache_fp32_v1_20260905`，同六条记录、同数值与父场哈希，仅 coarse_norm/truth_norm 存储改为 float32（static7 仍 f16）。CPU 消费者测试 20 项通过、独立只读审计 CLEAR 后 detached 执行：`cache_stage_fp32_v1/smoke_terminal.json`（SHA256 `e0a7518d06a8c4102fdfbb108cff12bb445df37244130fe20946c3ad4912cb9e`）status=complete，六条记录 E_q 全部精确为 0、逐字节回读一致、f16 位向下投影逐字节复现本 stage 的 f16 分片，用时 30.554 s、峰值 352,735,232 bytes。结论：f16 拒绝纯属存储表示损失。E_q=0 仅证明存储无损，不代表学习收益；完整 fp32 缓存（预计 49.468 GiB）与训练均未授权。详见 `cache_stage_fp32_v1/research_decision.md`。
