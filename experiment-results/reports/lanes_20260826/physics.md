## finding

1. **校正量是 feature,不是真值监督。** 预测的 rank-16 系数完全来自 29 个非真值通道(log v、∇log v、source、travel、坐标、k0/k1 两帧观测误差、**parent** 场的 POD 投影、route one-hot):`r16_dscp.py:236-257`。真值只出现在训练损失的目标端(`open_truth` → `exact_microblocked_loss`,`r16_dscp_engine_v4.py:83-84`、`r16_dscp_training_v2.py:113-172`)。因此正确标注是:**输出场对 truth 的精确监督(仅 train split),校正本身是特征驱动的低秩 ansatz;全流程零 PDE 残差**——没有 LWC-84 8阶空间/4阶时间算子、没有 CPML memory 变量、没有内部 dt=1e-5,只在 restriction 后的 201×201 / dt_out=2.5 ms 输出网格上做数据拟合(`configs/datasets/..._v5_dt1e5_t401.yaml:26-47`)。物理约束只保留两条且是自洽的:`correction[:,0,:]=0`(顶部 p(z=0)=0)与 C1 causal mask 在 k1 前恒零(`r16_dscp.py:106-122,371`),与零初始条件、左右下 CPML 不冲突。

2. **量纲可比,尺度不可比,但压制机制不是权重。** frame 与 late 都是同一无量纲量(逐帧相对 L2 平方,`r16_dscp.py:420-424`),故量纲一致。late 的等效逐帧权重仅 1.75×((1/T)Σ_all + 0.25·(3/T)Σ_late)。真正的主导来自分母:uniform late 逐帧相对误差 49.2 → f_t≈2.4e3,early 0.128 → f_t≈0.016,故该记录 total≈1767(`terminal.json:122,145`),>99% 来自 late/mid 比值项;把 0.25 改成 0 几乎不变。**跨族并未被 loss 量级压掉**:每步 `clip_grad_norm_(...,1.0)`+AdamW 已归一化步长(`r16_dscp_engine_v5.py:90-93`),3 记录 round-robin 等次等步长(`r16_dscp_engine_v8.py:37`)。

3. **基不是三族混用,故"必然反方向"不成立。** basis 为 [3,401,16] 每族独立,uniform 块由 2 条 uniform 记录的原始残差时间协方差 eigh 得到(`scripts/train_r16_dscp.py:430-441,604-637`)。实测 uniform correction_energy_ratio=7.5e-4(幅度 2.74%),aggregate 0.07599→0.08162;由此推得 cos(校正, parent 误差)≈+0.03,即**近正交噪声 + 极小的增误差投影,不是符号反转**;early/mid/late 三段同时变差(0.1280→0.1303、1.171→2.290、49.2→63.5)也符合"固定时间形状把幅度注入近零真值帧"。

4. **router 足够。** uniform 为常速(`lwc84_manifest.py:216-222`),fx=fz=0 精确成立 → `r16_dscp.py:90-92` 确定性判为 uniform;head 共享但 route one-hot 经 depthwise+SiLU+bias 可表达符号翻转。瓶颈不在 router。

**根因**:均匀介质 + 三边 CPML(npml40,R=1e-8)+ 顶部 Dirichlet + 零初始态,在 1.0 s 记录的后 1/3 场已基本逸出,||truth_t||→极小,而 parent 残留伪能量(late 4900%)。逐帧相对分母在近零帧上病态,训练项与门限量(`mean_frame_rel_l2`/`family_gain`,`r16_dscp_engine_v3.py:134-149,310-317`)同被该无物理意义比值支配。

## uncertainty
cos≈0.03 与 late 真值能量为推导量(假设 ‖p‖≈‖t‖),未直接测量逐帧真值能量;`late` 是否对全部 uniform 记录都近零未验证(仅 train_uniform_00321)。

## recommended_next_step
1. train-only 离线量:3 条 smoke 记录逐帧 ‖truth_t‖²/max_t 曲线;若 late<1e-3 则把逐帧分母换成带能量下限的记录级归一,再谈 late 权重。
2. 修 smoke `loss` 门:现值 0.9997 是 record0(uniform,1767)对 record191%3=2(marmousi,0.588)的跨记录伪迹(`r16_dscp_engine_v8.py:37`、`_engine_v3.py:320`),不能作为学习证据。
3. 用本族基在**本批记录**上重算 oracle 上界,替换 `_engine_v3.py:320` 硬编码 oracle(0.2225 等),再判 uniform 是否可达。

## veto_reason
- `normalized_coefficient_energy` 名实不符:实为 `coefficients.square().mean()`(`r16_dscp.py:431`),量纲为幅度²,与另三个无量纲比值直接相加 —— 量纲不一致,须改名或除以 field_scale²。
- 否决把该损失或校正表述为"物理/PDE 残差"或"部署期真值监督";它是输出场数据拟合 + 特征驱动 ansatz。
- 否决以 smoke `loss` 门通过作为收敛/学习证据。
