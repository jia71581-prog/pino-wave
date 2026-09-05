# v16 pilot fail_gate 报告(诚实记录)

日期:2026-08-26。预注册 `results/r16_dscp_v16_preregistration_20260826.json`(sha `d91689b4`)。
链日志 `results/r16_dscp_v16_auto_chain_20260826.log`;终端 `results/r16_dscp_v16/{smoke,pilot}/terminal.json`。
授权:lead 条件性指令"继续，符合条件启动长训"—— **pilot 门未过,长训按预注册未启动**。

## 链实况

- **smoke:passed**(129 s,峰值 1.54 GiB)。六门全过;oracle fraction 0.436/0.682/0.812
  与 r4e12 W 臂逐位一致 —— 确定性与管线保真完好。
- **pilot:fail_gate**(849 s,其中 bundle 构建 764 s)。24 pilot_fit 训练 3072 updates,
  24 held-out pilot_confirm 评估:
  - joint_improvement:**−0.0323**(门 ≥0.01)
  - per_family:uniform −0.0014 / layered −0.0003 / marmousi **−0.0952**(门 ≥0.005 全部)
  - nonworse:**10/24**(门 ≥23)
- 链在 pilot 停止,long 未启动。v16 记 **1 次 failed_gate**。

## 轨迹判读(每 384 updates 评一次 confirm)

| update | mean | nonworse | layered | marmousi | uniform |
|---|---|---|---|---|---|
| 384 | −0.0009 | 11/24 | **+0.0438** | −0.0427 | −0.0037 |
| 1152 | −0.0123 | 12/24 | +0.0281 | −0.0654 | +0.0005 |
| 3072 | −0.0323 | 10/24 | −0.0003 | **−0.0952** | −0.0014 |

三条结论:

1. **marmousi 是即时迁移失败**,首快照已 −0.043 且单调恶化,与训练时长无关。
2. layered 是标准过拟合(早期 +0.044 → 0);uniform 全程无信号(±0.003)。
3. **轨迹上不存在过门的点**:早停救不了 pilot,失败不是预算/日程问题。

## 科学定位(与探针链的关系,如实)

r4e10-r4e12 证明的是**每记录拟合容量**:头在 round-robin 的记录上能恢复接近各记录
oracle 的校正(r4e12 smoke 本质上是 3 记录记忆)。r4e11 C2 的"特��信息充分"同样是
每记录意义。pilot 首次测试**同一组权重跨记录泛化**,结果为否(24 fit 记录规模下)。
探针链结论不被推翻,但其声明边界("非泛化证据")到此兑现。

## 待归因的实现注记(诚实披露)

pilot 从 smoke last.pt **热启动**(auto 链的实现选择,预注册 stages 未写明热启动)。
3 记录记忆可能污染跨记录学习。r4e13 冷启动对照臂将分离这一因素;在对照落地前,
"wide128 跨记录不泛化"的结论只在"热启动+24 fit 记录"条件下成立。

## 下一步(r4e13 探针,已冻结)

冷启动对照 + 数据规模阶梯(24/96/192 fit 记录,均评同一 24 confirm),直接回答:
(a) 热启动是否是损害因素;(b) 泛化是否随 fit 记录数上升 —— 即"192 记录的 long
能否救"这一问题本身,而不是靠再跑一次 long 去赌。
