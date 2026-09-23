# 论文修订记录与贡献—证据对照（2026-09-22）

本次按 manuscript-optimizer → scientific-writing → write-scientific-manuscript → scientific-prose-style 修订，并使用 citation-verifier、claim-source-verification 与 pdf 检查。不是通过文字把诊断研究包装成精度突破。

## 当前主张

论文题目改为 **Diagnosing Late-Time Accuracy Limits of Single-Shot Acoustic Neural Operators in Heterogeneous Media**。核心贡献是把晚期误差定位、有限校正族的条件界、监督覆盖和控制实验连成可复核的研究程序。标准恒等式与 background+correction 结构本身不声称原创。

| 可声称的贡献或效果 | 当前证据 | 必须同时说明的边界 |
|---|---|---|
| 复杂介质晚期误差定位 | anchor M96 B3=0.572774；晚段能量30.61%/误差64.10% | train panel，0–1s内后段，非>1s外推 |
| 对固定校正方向作定量诊断 | corrected R14 deployed scalar oracle=0.999234；144帧、12记录 | 每帧使用真值拟合；不限制重新训练或其他结构 |
| 时间监督迁移与覆盖问题 | 单记录16→51探针；V3有限访问覆盖 | 单例效果不等于全验证改善；V3不是anchor真实16帧continuous调度 |
| 透明的控制实验 | R4 18,700步后M-B3 +2.83%，训练loss下降 | R4=16帧；train panel；只能限制此配方/预算 |
| 实际查询加速 | standard3.97s=2.5–5.1x；tb16=2.54s | tb16整场一致待核；IC获取/训练不计入；误差更高 |

## 实质更正

1. “single-shot不存在误差累积”改为无自回归反馈；长时幅相误差仍可增长。
2. “八/九种解释完全排除、两机制已确证”改为具体实验限制和诊断解释；容量、物理载波、attention、残差训练不作全局否定。
3. 0.3169由均值相乘得到，不是精确record-equal恒等式。既有scalar重算为0.312137，pooled0.287059；公式、标量源和脚本均加入诊断目录。
4. 旧alignment probe漏乘自由表面输出因子，并通过浮点相减提取小分支。正文改用已有R14门前直接捕获与deployed空间，ratio0.999234。旧0.9984不再作部署界。原P1半谱未作Hermitian权重，故不能把约49%称为物理L2精确幅相份额。
5. PDE A(error)不是独立无标签约束，但可能改变优化；loss小不等于梯度小。移除“provably uninformative”和“提高权重不能有效”。
6. “频谱flattening至多23.5%”改为最低频带固定时的条件反事实；它不是新方法的总上限或已实现改善。
7. 修正train frozen-panel误写held-out、A4与anchor混用风险、R4误写51帧。test_id sealed与Marmousi平移twins保留；10%诊断阈值与5%/10x工程目标区分。
8. 运行比较明确输出匹配而输入信息/计时范围不同；“same outputs”改为“not re-evaluated”，细网格误差改为includes resampling。
9. 低能量频带不自动等同数值病态；需要绝对能量和噪声检查。吸收边界不应以全域严格能量守恒表述。
10. 高相关不能证明损失项梯度冗余；单次冻结gate sweep不能证明联合训练全局最优。

## 文献与写作

本轮在线核对9个arXiv条目的存在、题目和作者，补全5个作者缺失条目；现有2个PNAS DOI未重新做完整出版元数据审计。LaTeX 11个引用键均有对应条目，没有placeholder。

4类过强文献映射被移除或收窄：把Balaji分阶段训练断言为成功唯一原因；把attention差异当因果解释；把PDE-Refiner的rollout机制直接当作本项目机制复现；把TF-SNO动机当作本解码器缺陷已经确证。保留文献真正支持的方法动机与问题范围。相关原文：[Balaji](https://arxiv.org/html/2602.11197v1)、[PDE-Refiner](https://arxiv.org/abs/2308.05732)、[TF-SNO](https://arxiv.org/abs/2606.21189)。这不是完整逐句引用审计。

摘要、引言和结论统一为问题—诊断方法—已测效果—适用边界，移除自我贬低式开场，同时不加入未经实验的收益。两张既有图未修改。历史01–07及旧台账保留，不要求它们在本轮全部同步；与本记录冲突的旧解释不应再次引用。

## 保留与验证

原文、PDF与两级README已保存在 revisions/20260922_before_optimization/，MANIFEST.json记录原始SHA-256。没有修改实验代码、配置、检查点或原图。新results/diagnostic_audit_20260922只复制已有R14和scalar证据，新增CPU标量重算脚本及输出。

构建、标量重算、包验证与视觉检查的最终结果见 notes/verification_20260922.json。PDF为14页，检查全页缩略图及运行表/校正公式页。报告未声称新训练完成、独立泛化成立或目标达标。
