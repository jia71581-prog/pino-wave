# r4e11 探针规格:特征头容量/信息阶梯(冻结于运行前)

日期:2026-08-26。类别:**train-only 诊断探针,非 stage,非晋级候选**。
授权背景:lead 以"请自行推进"(2026-08-26)委托本会话自主推进;本探针仍遵守全部在册 veto,
不晋级、不动阈值、不触 validation/test_id。

## 问题

r4e10 已证明:直拟臂(绕过特征头)达 oracle (iii) 上界的 80–99%,而 1202 参数头
在 4× smoke 预算下最高 30%。但 r4e10 **没有**区分:

- **H-C(容量)**:29 通道特征里有足够信息,只是 1202 参数/5×5 感受野的头表达不了;
- **H-D(信息)**:特征本身不含恢复 oracle 系数所需的信息,加多大的头都没用;
- **H-E(损失面)**:field 损失的优化面病态,监督目标换成 oracle 系数后同一个头就能学。

三者对应完全不同的 v16 设计,必须先分离。

## 绑定(启动前逐一复核,漂移即拒)

继承 r4e10 规格的全部 9 项绑定(`results/r4e10_optimization_gap_spec_20260826.md`
sha `0c2739a561da8595cbdeb5c020abbf84adcaafe80ed9ba4ad4ed17618824f6ff`),另加:

| 文件 | sha256 |
|---|---|
| `scripts/probe_r4e10_optimization_gap.py`(管线复用) | 8ea80e3327b206534430cee7fc209261f22bc215a0944a75d544f2c999e1d0b9 |
| `results/r4e10_optimization_gap_20260826/leg_a/terminal.json`(只读参照) | 7b22fa2418fadd3266fa9e668b1f0bdc1bc06241f23c84267a34c55e94c23c2d |
| `results/r4e10_optimization_gap_20260826/leg_b/terminal.json`(只读参照) | 38e83546d269c285349bdb00b28bc417cbffd072cf74ac2e0de4141a544aa5f4 |

## 共享设定

- 管线、记录、保真门与 r4e10 完全一致(复用其 `build_pipeline` / 保真函数;
  保真任一不过 = invalid_pipeline 作废)。
- **oracle 系数目标 â**:与 v15 oracle 同一加权 LS(parent 掩码保留帧,truth 能量下限权重),
  逐像素 [16,H,W];回归目标用 â_clip = clamp(â, ±0.999·scale)(可表示域内),
  评估一律走同一 confined 校正路径算 (iii) achieved gain。
- 所有头输出层均为 tanh·scale(与 v15 同参数化),特征仍为冻结的 29 通道
  deployment-causal 集合;**不改 basis、不改 rank、不加空间谱模态**。
- 优化器 AdamW(0.9,0.99)/1e-8/wd 1e-4/clip 1.0,lr 3e-3(r4e10 最优臂),
  seed 372,每训练臂 768 updates round-robin(与 r4e10 leg A 同预算),
  并在 192 updates(smoke 等效)处留快照。

## 臂定义

| 臂 | 参数量级 | 监督 | 回答什么 |
|---|---|---|---|
| R(闭式 ridge) | 480(线性 1×1) | â_clip 闭式加权最小二乘 | 逐像素线性信息下限 |
| M1t(v15 头,目标回归) | 1202 | min Σ((c−â_clip)/scale)² | H-E:换监督后原头能否学 |
| M2f(加宽头,field 损失) | ~9k | v15 masked_confined_loss | H-C:容量(局部感受野) |
| M2t(加宽头,目标回归) | ~9k | 同 M1t | 容量 × 监督交互 |
| M3f(扩感受野头,field 损失) | ~9k | v15 masked_confined_loss | H-C:非局部信息 |
| M3t(扩感受野头,目标回归) | ~9k | 同 M1t | 同上 × 监督 |

- M2 = depthwise-separable 29→64→64→16,3×3 全 dilation1(感受野 ~7px)。
- M3 = 同宽度,三段 depthwise dilation 1/4/16(感受野 ~43px)。
- 实测参数量、单次前向时延写入 terminal(v16 须过 adapter_mean 0.35 s 与 checkpoint 2 MiB 门,
  预检:~9k 参数 fp32 = 36 KB,前向远小于现 0.025 s 量级,不构成风险)。
- 目标回归臂同时报**拟合质量** fit_rel_err = ||c−â_clip||/||â_clip||(逐记录):
  这是信息瓶颈的直接读数。

## 预注册判读

f = achieved (iii) gain / 本记录 oracle 上界(768 updates 终点):

- (C1 容量确认)某个可部署臂(M2*/M3*)在 ≥2/3 记录 f ≥ 0.5 → **H-C 成立**,
  v16 = 该头族(按最优臂定型),进入 v16 预注册起草。
- (C2 信息瓶颈)所有共享头臂在 ≥2 记录 f < 0.5,且最优目标回归臂 fit_rel_err > 0.9
  (即使换监督、加容量、扩感受野也几乎拟合不了 â)→ **H-D 成立**,
  v16 方向改为特征增强(新 deployment-causal 通道设计,另立探针)。
- (C3 损失面)M1t 的 f 显著超过 r4e10 leg A lr3e-3 臂(≥2 记录 f 提升 ≥0.15)
  而 M2f/M3f 相对 M2t/M3t 无同等优势 → **H-E 参与**,v16 须改训练目标
  (两阶段:先回归 â 预训练,后 field 损失微调,设计另议)。
- (C4)其余组合 → 按记录报告,基于最大 f 臂选后续探针,不下单一结论。
- C1 与 C3 可同时成立(容量+监督都重要);判读按数值如实并列。

## 预算与安全

- 单 GPU(GPU 0),总 GPU ≤ 1200 s(估计 ~5 min:6 臂 × ~25 s + 目标计算 + 保真);
  peak reserved ≤ 20 GiB;磁盘启动前 ≥ 2 GiB 空余;产物仅 JSON ≤ 32 MiB。
- 写入白名单:`results/r4e11_head_capacity_ladder_20260826/` 新建;
  零修改既有文件、零删除、零模型检查点落盘。
- 父检查点 sha 前后各复算一次;validation/test_id 不可触。

## 与已证伪清单的区分(逐条)

- "高频空间头":该方向预测场的高频空间残差;本探针所有头仍只预测 rank-16 POD
  时间基系数图,输出空间与 v15 相同。
- "CNN 残差 meta-adapter(r4 parent)":该方向用 CNN 场到场直接改 parent 残差;
  本探针头输出经固定时间基展开 + confined 掩码,不做场到场映射。
- "加空间谱模态/rank":本探针不改 basis、不改 rank(rank 留量另案)。
- "onset 两帧驱动整段未来修正":onset 两帧误差只是 29 通道中的 3 个,
  且该通道集为 v15 已冻结特征,本探针不新增 onset 依赖。

## 声明边界

- 3 记录上的拟合容量测试,**不是泛化证据**;C1 成立也只支持"进入 v16 预注册起草",
  v16 的 pilot/long 泛化门照旧。
- oracle 与 â 均读 train 真值,是上界/目标参照,不可部署、不是达成结果。
- 不动任何阈值(veto a);不晋级(veto f 类推);校正非物理/PDE 残差(veto e)。
