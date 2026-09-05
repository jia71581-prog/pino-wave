# r4e12 探针报告:M2 宽头收敛地平线

日期:2026-08-26。规格:`results/r4e12_convergence_horizon_spec_20260826.md`(sha `cfb46492`,冻结于运行前)。
产物:`results/r4e12_convergence_horizon_20260826/terminal.json`(status=success)。
运行 211.6 s,峰值 reserved 7.73 GiB,父检查点 `448035bd` 前后一致,validation/test_id 未开封,
零检查点写入,阈值零移动,绑定全部零漂移。保真门:uniform 首损失与 v15 smoke 逐位一致,
oracle (iii) 上界 rel diff = 0。

运行注记:首次启动因未设 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 在保真门内即被 determinism
检查拒绝(零训练步、零写入),补环境变量后完整重跑;两次启动均无 terminal 残留冲突。

## 冻结判读:**D1 成立**(头族+预算充分),W+ 并列成立

f = achieved (iii) gain / 本记录 oracle 上界,3072 updates 终点:

| 臂 | 参数 | 配置 | uniform | layered | marmousi | D1 |
|---|---|---|---|---|---|---|
| L | 8690 | M2/恒定 lr/field | +0.32 | +0.45 | +0.66 | ✗(1/3) |
| S | 8690 | M2/cosine/field | +0.32 | +0.46 | +0.64 | ✗(1/3) |
| **W** | **25266** | **M2 宽 128/恒定/field** | **+0.44** | **+0.68** | **+0.81** | **✓(2/3)** |
| H | 8690 | 回归预训 768+field | +0.31 | +0.51 | +0.64 | ✓(2/3) |

对照:v15 头(1202 参,r4e10 最优臂)= 0.095/0.062/0.302;r4e11 同结构 768 upd:M2f = 0.24/0.31/0.55。

## 实证结论

1. **容量阶梯三级全程单调**:1202 → 8690 → 25266 参,f 三记录全部随宽度上升;
   W−L 宽度边际 +0.12/+0.24/+0.15,全过 0.05 线(**W+**)。r4e10 定位的头瓶颈
   在 25k 尺度仍未耗尽:W 在 3072 终点仍在爬(layered +0.032/最后 768 upd)。
2. **预算是 r4e11 C1 未过线的主因**:同一 8690 头同一配置,768→3072 updates 把
   f 从 0.24/0.31/0.55 抬到 0.32/0.45/0.66 —— r4e11 判 C4 而非 C1 是截断伪象。
3. **cosine 日程无收益**(S≈L,差 ≤0.02):恒定 3e-3 在此头族不构成障碍,
   v16 不需要日程复杂度。
4. **混合监督只在 layered 有小幅收益**(H vs L:layered +0.06,uniform/marmousi ≈0):
   与 r4e11 结论 3(â_clip 回归对高饱和 uniform 有害、对低饱和 layered 有益)一致;
   收益不及加宽的一半,v16 主推荐不含混合监督。
5. **绝对量级**:W 的 achieved gain (iii) = uniform +0.054(0.0760→0.0719)/
   layered +0.143(0.1051→0.0900)/ marmousi +0.108(0.6014→0.5365)。
   layered 训练记录被拉进 0.10 线内;marmousi 的 parent 本身太差(0.60),
   校正天花板受 oracle 上界(0.133)限制,这是 basis/rank 问题,不是头问题。

## v16 定型建议(证据链:r4e10 → r4e11 → r4e12)

- 结构:M2 拓扑宽 128(25266 参,depthwise-separable,3×3/dilation1),tanh·scale
  输出契约不变,confined 校正路径不变,29 通道冻结特征不变。
- 监督:field(masked_confined_loss)单目标;lr 3e-3 恒定;预算 ≥3072 updates/记录集
  (W 在 3072 仍爬,v16 预注册的预算与可达性阈值须据本探针轨迹重估并披露)。
- 部署代价:25k 参 fp32 ≈ 100 KB,前向仍亚毫秒,距 0.35 s/2 MiB 门余量巨大。
- 遗留风险如实声明:uniform f=0.44 未过 0.5(但其 oracle 上界本就最小,0.124,
  绝对 gain +0.054 与其他族同量级);3 记录 train-only,泛化未证,pilot 门照旧。

## 声明边界

train-only 收敛/容量测试,非泛化证据;oracle/â 读 train 真值仅作参照;探针非 stage,
不占 v15 failure 记账,不晋级;已证伪清单不复活(头仍只输出 rank-16 时间基系数图)。
