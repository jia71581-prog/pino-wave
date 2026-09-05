# r4e12 探针规格:M2 宽头收敛地平线(冻结于运行前)

日期:2026-08-26。类别:**train-only 诊断探针,非 stage,非晋级候选**。
授权背景:lead"请自行推进"(2026-08-26)持续有效;遵守全部在册 veto,
不晋级、不动已冻结阈值、不触 validation/test_id、不写模型检查点。

## 问题

r4e11 落 C4:8690 参数 M2/M3 头把 f(= achieved (iii) gain / 本记录 oracle 上界)从
v15 头的 0.06~0.30 抬到 0.21~0.59,但无臂在 ≥2/3 记录达 0.5;同时六个可部署臂在
最后 576 updates 仍爬 +0.12~+0.28,**远未平台**。C1 未过线的最可疑解释是预算截断,
而非容量上限。本探针测四件事:

- **平台位置**:M2f(field 损失)在 4× 预算(3072 updates)下的收敛地平线;
- **lr 日程**:恒定 3e-3 的震荡是否掩盖平台(cosine 衰减对照);
- **容量边际**:宽度 64→128(~9k→~30k)的边际收益是否仍为正;
- **混合监督**:â_clip 回归预训练(768)+ field 微调(2304)是否优于纯 field
  (r4e11 结论 3:回归监督在低饱和族最优、uniform 有害;混合能否取二者之长)。

## 绑定(启动前逐一复核,漂移即拒)

继承 r4e10 `verify_bindings()` 全部 9 项,另加:

| 文件 | sha256 |
|---|---|
| `scripts/probe_r4e10_optimization_gap.py` | 8ea80e3327b206534430cee7fc209261f22bc215a0944a75d544f2c999e1d0b9 |
| `scripts/probe_r4e11_head_capacity_ladder.py` | RUNTIME_PIN_R4E11_SCRIPT |
| `results/r4e11_head_capacity_ladder_20260826/terminal.json` | RUNTIME_PIN_R4E11_TERMINAL |
| `results/r4e10_optimization_gap_20260826/leg_a/terminal.json` | 7b22fa2418fadd3266fa9e668b1f0bdc1bc06241f23c84267a34c55e94c23c2d |

(RUNTIME_PIN_*:脚本启动时首次读取并写入 terminal 的 bindings 表;
本规格冻结其**引用关系**,数值以启动时实测为准,启动后再漂移即拒。
r4e10 的 9 项含父检查点/basis/panels/引擎源码,均有硬编码期望值。)

## 共享设定

- 管线、3 条 smoke 记录、保真门(pipeline_fidelity + initial_loss_fidelity)
  与 r4e10/r4e11 完全一致;保真任一不过 = invalid_pipeline 作废。
- 头输出契约与 v15 相同:tanh × 冻结 per-family scale,confined 校正路径,
  29 通道冻结特征;**不改 basis、不改 rank、不加谱模态**。
- AdamW(0.9,0.99)/eps1e-8/wd1e-4/clip1.0,seed 372,round-robin 3 记录,
  3072 updates/臂,每 384 updates 全记录评估一次(f 轨迹)。
- oracle 系数目标 â_clip(仅 H 臂用):r4e11 同一函数。

## 臂定义(全部可部署结构)

| 臂 | 结构 | 监督 | lr |
|---|---|---|---|
| L | M2(64 宽,8690 参) | field(masked_confined_loss) | 3e-3 恒定 |
| S | M2 同 L | field | cosine 3e-3→3e-5(T=3072) |
| W | M2 加宽至 128(参数量实测入册,~30k) | field | 3e-3 恒定 |
| H | M2 同 L | 前 768 â_clip 回归 + 后 2304 field | 3e-3 恒定(切换时重置优化器) |

## 预注册判读

f_final = 3072 终点;"仍在爬" := f(3072) − f(2304) ≥ 0.02(任一记录)。

- **(D1 达标)** 某臂 f_final ≥ 0.5 于 ≥2/3 记录 → 头族+预算充分,
  v16 = 该臂配置,进入 v16 预注册起草(预算按其轨迹达 95% 平台的 update 数定)。
- **(D2 平台低于线)** 所有臂全记录均不在爬,且无臂过 D1 → 9k~30k 局部卷积头族
  的平台 < 0.5:按记录报平台值,v16 可达性目标必须按实测平台重估(在 v16
  预注册中披露本探针;不动 v15 已冻结门,veto (a) 不涉及新预注册的先验重估)。
- **(D3 未收敛)** 最优臂(mean f 最大的可部署臂)仍在爬且未过 D1 →
  **预授权单次延长**:该臂再训 3072(总 6144),重判 D1/D2;延长后仍在爬则如实报
  "地平线 >6144",不再延长。
- **(D4 混合)** 其余 → 按记录报告。D1 与"W 边际收益 ≥ +0.05(≥2 记录)"可并列成立,
  后者单独记为 **W+**(v16 取宽版)。

## 预算与安全

- 单 GPU(GPU 0),总 GPU ≤ 1500 s(估计:4 臂 × 3072 × ~12 ms ≈ 150 s + 保真
  + 可能的延长 ~40 s);peak reserved ≤ 20 GiB;启动前磁盘 ≥ 2 GiB。
- 写入白名单:`results/r4e12_convergence_horizon_20260826/` 新建;零修改既有文件、
  零删除、零模型检查点落盘(评估后即弃)。
- 父检查点 sha 前后各复算;validation/test_id 不可触;阈值零移动。

## 与已证伪清单的区分

同 r4e11 全部四条(头仍只输出 rank-16 时间基系数图;不做场到场;不改 rank/谱模态;
不新增 onset 依赖)。H 臂的回归预训练用 train 真值导出的 â_clip,与 v15 训练同一
信息面(train 期真值可用),部署期不需要真值 —— 不违反 veto (b)。

## 声明边界

3 记录 train-only 拟合容量/收敛测试,非泛化证据;不占 v15 failure 记账;
D1 成立也只支持"进入 v16 预注册起草",pilot/long 泛化门照旧。
