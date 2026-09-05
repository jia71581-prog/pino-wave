# v19 面板外一次性确认报告(2026-08-27)

prereg sha 375ce27c(评估前冻结,一次性,无重试)。评估集 final_train_confirm 24
(8/8/8 族均衡,与 fit2/long_calibration 零重叠,首次使用)。terminal
results/r16_dscp_v19_offpanel/terminal.json(success,parent 未动,sealed 未开)。

## 判读(按冻结标准:joint ≥ 0.7×校准值 且 tol ≥22/24 且 worst ≥ −0.02)

| | 校准 joint | 面板外 joint | 保持率 | tol@1% | worst | 判读 |
|---|---|---|---|---|---|---|
| A_data | +0.0320 | +0.0213 | 67% | 21/24 | −0.0353 | **INCONSISTENT** |
| B_data_hinge | +0.0363 | **+0.0265** | 73% | 23/24 | −0.0116 | **CONSISTENT** |

- A 臂三处未达:joint 差 0.0011(0.0213 vs 门 0.0224)、tol 21<22、
  worst −0.0353 破 −0.02 底线(train_layered_00848)。
- B 臂全过:hinge 的危害抑制在面板外**成立**(超容差仅 1 条 −0.0116,
  A 臂 3 条最深 −0.0353)。B 面板外仍优于 v16/v17 的校准面板成绩(+0.0203)。

## 结论

1. **B(+hinge)是该线的可交付配置**:面板外保持 73% 增益、尾部受控,
   偏好顺位 B > A 现在有面板外证据支撑。
2. A 的 INCONSISTENT 如实落账:纯数据臂对校准面板存在轻度过拟合,
   hinge 是弥合面板内外差距的那个组件(两臂唯一差异)。
3. 数字如实:面板外增益普遍低于校准面板(A 67%/B 73% 保持率),
   校准面板数字不应被引用为泛化声明;论文引用应以 v19 面板外数字为准。

## 记账

一次性阶段按预注册关闭,无重试。0.05 验收主张仍需 sealed validation
(lead 明示解封后一次性评估,评估对象建议 = B_data_hinge/best.pt)。
