# r4e11 探针报告:特征头容量/信息阶梯

日期:2026-08-26。规格:`results/r4e11_head_capacity_ladder_spec_20260826.md`(sha `2e01ef64`,冻结于运行前)。
产物:`results/r4e11_head_capacity_ladder_20260826/terminal.json`(status=success)。
运行 80.0 s,峰值 reserved 7.73 GiB,父检查点 `448035bd` 前后一致,validation/test_id 未开封,
零检查点写入,阈值零移动。全部绑定(r4e10 的 9 项 + 新增 3 项)零漂移。

## 管线保真(先于一切解读)

- uniform 首损失与 v15 smoke **逐位一致**(rel diff 0.0);layered/marmousi 2.2e-5 / 4.2e-7。
- parent_rel_l2 rel diff ≤ 1.6e-12;oracle (iii) 上界三记录 rel diff = **0.0**。

## 冻结判读:落 C4(混合),三个单一假设均未按预注册规则成立

- **C1(容量)未过,但只差一格**:无可部署臂在 ≥2/3 记录达 f≥0.5。最接近的是
  M3t layered 0.684 + marmousi 0.353,与 M3f marmousi 0.593 + layered 0.332 —— 各臂都只有 1 条记录过线。
- **C2(信息瓶颈)被否**:best fit_rel_err = **0.346**(M3t uniform)远低于 0.9。
  29 通道特征**确实携带**恢复 oracle 系数图的信息 —— H-D 不成立,v16 不需要特征增强。
- **C3(损失面)被否**:M1t(同一 1202 参数头,换 oracle 系数回归监督)相对 r4e10 field-loss
  lr3e-3 臂在 marmousi −0.29 / uniform −0.38,只有 layered +0.014。换监督救不了小头。

## 数值(f = achieved (iii) gain / 本记录 oracle 上界,768 updates 终点)

| 臂 | 参数 | 监督 | uniform | layered | marmousi |
|---|---|---|---|---|---|
| R(闭式 ridge) | 480 | â_clip 闭式 LS | −10.29 | −1.81 | +0.15 |
| M1t(v15 头) | 1202 | â_clip 回归 | −0.28 | +0.08 | +0.01 |
| M2f(宽头) | 8690 | field 损失 | **+0.24** | +0.31 | +0.55 |
| M2t(宽头) | 8690 | â_clip 回归 | −0.27 | +0.65 | +0.39 |
| M3f(dilated) | 8690 | field 损失 | +0.21 | +0.33 | **+0.59** |
| M3t(dilated) | 8690 | â_clip 回归 | −0.15 | **+0.68** | +0.35 |

参照:r4e10 v15 头 field 损失最优臂(lr3e-3)= +0.095 / +0.062 / +0.302;直拟臂 = 0.80 / 0.93 / 0.99。

## 四条实证结论

1. **容量是真瓶颈的主成分**:7.2× 参数(1202→8690)在同损失同预算下把三记录全部抬升
   (uniform +0.095→+0.24,layered +0.062→+0.33,marmousi +0.302→+0.59),且 M2f/M3f
   三记录全正、无一退化。方向与 r4e10 的头容量定位一致,幅度尚未达 C1 的 0.5 线。
2. **768 updates 远未收敛**:六个可部署臂在最后 576 updates 仍爬 +0.12~+0.28
   (对照 M1t 已平台化的 +0.002~0.004)。9k 头的平台位置未知 —— 这是 C1 未过线的
   最可疑解释,也是下一探针要测的第一件事。
3. **â_clip 回归监督对 uniform 系统性有害,机制已定位**:M2t/M3t uniform 拟合很好
   (fit_rel_err 0.39/0.35)但 gain 为负(−0.27/−0.15),而同头 field 损失 +0.24/+0.21。
   uniform 的 oracle 系数 9.8% 像素饱和(max|â|/scale=101,r4e10 实测),clamp 后的
   â_clip **不是** tanh 可表示域内的场最优解(直拟臂证明域内存在 f=0.80 的解);
   拟合 â_clip 拟合得再好也不对齐场目标。layered/marmousi 饱和 ~0.1%/0.2%,故回归监督
   在 layered 反而最优(+0.68)。**推论:â_clip 回归可作低饱和族的辅助/预训练信号,
   不可作 uniform 的唯一监督;field 损失是跨族稳健的主目标。**
4. **~43px 感受野相对 ~7px 无决定性优势**(M3 vs M2 同监督差 ≤0.06,方向不一):
   非局部空间信息在此尺度不是短板,v16 不必为感受野付结构代价。
5. (旁证)480 参线性 ridge 在 uniform/layered 灾难性为负 —— 特征→系数映射
   实质非线性,逐像素线性下限彻底排除。

## v16 方向裁定(证据支持,待 lead 确认)

v16 = **M2 族宽头(depthwise-separable,~9k 参,3×3/dilation1)+ field 损失主目标**,
预算需先由收敛地平线探针(r4e12)定:9k 头在 field 损失下的平台位置与更宽一档
(~33k)的边际收益。若平台 f≥0.5 可达,v16 预注册按新可达性证据披露重估
(veto (a) 只禁止用结果调门,不禁止在新预注册里基于探针披露重估;v15 的 0.80/50%
继承阈值已被 v15 smoke 证实在 192-update 预算不可达,该实测必须在 v16 预注册中引用)。

部署代价预检:M2 前向 0.44 ms(v15 头 0.32 ms),fp32 36 KB,距 adapter_mean 0.35 s
与 checkpoint 2 MiB 门余量巨大。

## 声明边界

3 记录 train-only 拟合容量测试,非泛化证据;oracle/â 读 train 真值,仅作上界与目标参照;
探针非 stage,不占 v15 failure 记账,不晋级;已证伪清单(高频空间头、CNN 场到场
meta-adapter、加 rank/谱模态等)不复活 —— 本探针所有头仍输出 rank-16 时间基系数图。
