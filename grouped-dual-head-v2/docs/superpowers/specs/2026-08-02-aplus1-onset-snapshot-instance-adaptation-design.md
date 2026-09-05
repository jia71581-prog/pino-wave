# A+1 父模型上的播前快照实例化微调 — 设计

2026-08-02. 承 A+1 混合求解器预训练确立(G2 记忆 0.065 / G3 泛化 sigma=2 layered 0.094 / marmousi 0.046,
两非平凡族充分高)+ 高阶 LWC PDE loss 升级完成。用户要求:利用"播前"(早期起振)波场快照做高效实例化微调,
进一步提高精度;微调 PDE loss 用高阶 LWC(已落地 time_order=4)。

## 关键转折(为何现在值得重做实例化微调)

之前的实例化微调工作(见 memory `fno-acoustic-instance-adaptation`)在**旧 coarse-MIONet 父模型**上做,
裁定是"信息墙 ∩ 结构墙":严格 2 早期帧信息量太少,连无限容量校正器都超不过冻结父场;CEILING 研究证明
K~48 帧稀疏观测才能压到 0.06-0.14。那时父模型本身有结构瓶颈(coarse 场对均匀锐波前 + 晚期帧表征上限)。

**现在父模型 = A+1**:可泛化的真实算子,held-out layered 0.094 / marmousi 0.046。父场已经很好,
微调要解决的不再是"修复结构性缺陷",而是"用新记录的早期快照把这个已很好的算子再校准到该实例"。
这是一个不同的、更有希望的问题设定 —— 残差已经小(A+1 residual 是低秩散射),早期快照提供的是
"这个具体速度/源的实际波前",可锚定 A+1 的低秩散射预测。

## 架构(复用已验证资产,单因素接 A+1)

现有 `OnsetAdaptedV5`(adapters.py)是**父模型无关**的:`raw_wavefield(parent_field, ...)` 只做
`parent_field + residual_gate * onset_conditioned_correction`,不依赖父场如何产生。所以接 A+1 = 把
`parent_field` 换成 A+1 模型的输出场。

- **父场来源**:A+1 模型 = `_model(base, manifest, ProbeVariant(local_field_helmholtz_synthesis=True,
  rank=8, ...))` 前向 + 背景 P_bg(部署时对新速度跑 sigma=2 平滑 solve)。A+1 输出场 =
  decode(encode(P_bg) + synthesis)。这天然契合实例化部署:新记录本就要跑一次 P_bg solve。
- **微调层**:保留 33 参 LoRA(`latent_delta` + `residual_gate` 安全阀),元学习超网络学"看早期快照→出补偿"。
- **因果契约**:严守 `future_truth_used=False`,只读 K=2 早期起振帧(`SnapshotAccessAudit` 强制)。
- **安全阀**:`residual_gate` 回滚门,微调不达标退化纯 A+1 父场(绝不恶化)。

## 微调损失(高阶 LWC 已就绪)

`instance_adaptation/losses.py::lwc84_residual(time_order=4)` 已升级:8 阶空间 `_laplacian8` +
4 阶 LWC 时间 modified-equation 项 `dt²/12·L²(p)`。实证:真值场 4 阶残差 0.0018 vs 2 阶 0.0079(小 4.4×),
即高阶不把时间截断误差当伪物理残差惩罚。微调 loss = observed(整场能量归一,K 早期帧) + w·高阶 LWC 自归一
PDE(第 2 观测帧后所有时刻,RAD 自适应采样集中高残差格) + 1e-5 anchor。全程在归一化 O(1) 空间(避免
decode 后 1e-9 病态,memory 已确证的根因)。

## 待验证的核心问题(G3 held-out 判据)

旧父模型下"2 早期帧超不过父场"是信息墙。**A+1 父模型下重新测**:
1. 微调后 held-out 是否 < A+1 冻结父场(sigma=2 layered 0.094 / marmousi 0.046)?
2. 若 2 帧仍不够(信息墙可能依旧),放宽 K 到含中/晚期稀疏帧(CEILING 研究指 K~48 可破),
   在线同化场景合理。
3. 高阶 LWC PDE 项在 A+1 残差上是否比 2 阶提供更有效的未观测时刻外推监督(高阶消除频散伪残差,
   理论上对波前位置/相位的物理约束更准)。

## 诚实边界

- A+1 部署本就需 P_bg solve(sigma=2,CFL 严),实例化微调叠加在此之上,不增额外 solve。
- 若信息墙在 A+1 下依旧(2 帧信息不足),价值在放宽观测契约或高阶 PDE 外推;安全契约保证绝不恶化。
- 缩减池 A+1 父模型(N=192 best ckpt)可作微调父场做首轮 pilot;正式需全池 A+1 或接受缩减池父场的
  held-out 基线。

## 实现步骤

1. N=192 sigma=2 G3 完成 → 取 best checkpoint 作 A+1 父模型(或用 N=48 sigma=2 ckpt,更快)。
2. 写 A+1 父场适配器接线:`run_v5_instance_adaptation` 的 parent 换成 A+1 model + P_bg provider。
3. 元超网络在 train-split A+1 残差上离线元训练(家族均衡 episode)。
4. G3 held-out pilot:因果 2 帧 + 高阶 LWC PDE,对照 A+1 冻结父场;记录是否改善 + 因果审计。
5. 若 2 帧不够,K=48 稀疏采样消融(直击 CEILING 研究指出的信息充足点)。
