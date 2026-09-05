# r16_dscp_v15 smoke 运行记录 — fail_gate

日期:2026-08-26。授权:`results/r16_dscp_v15/authorizations/smoke.json`(含 post_hoc_rule_revision 确认)。
终端产物:`results/r16_dscp_v15/smoke/terminal.json`(status = fail_gate)。

## 结果

192 updates(每记录 64 次观测),总耗时 136.7 s(预算 600 s 内),peak VRAM 6.30 GiB
(torch.cuda.max_memory_reserved,非平凡测量,8 GiB 门通过)。父检查点写审计:writes=0,
sha256 前后均为 448035bd,read_only_unchanged。validation / test_id 未开封。

三道科学门中两道失败,nonworse 门失败:

| 门 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| per-record loss reduction | >= 0.80(每记录) | uniform -0.0014 / layered 0.2562 / marmousi 0.0830 | **fail(3/3 记录)** |
| oracle_gain(口径 iii,>=50% 本记录重算上界) | 0.0622 / 0.1051 / 0.0665 | uniform **-0.0068** / layered 0.0062 / marmousi 0.0327 | **fail(3/3 记录)** |
| nonworse | 3/3 | 2/3(uniform 变差:0.07599 -> 0.07651) | **fail** |
| VRAM(train) | < 8 GiB,非平凡 | 6.30 GiB 已测 | pass |
| finite / space | — | — | pass |

oracle 上界本身为正且全部在被评分记录上重算(0.1244 / 0.2102 / 0.1329),
v14 三个作废常数经断言不在任何 v15 非测试源中。

## 诚实读法

1. **fail_gate 是四项 v14 缺陷修复生效后的第一读数。** v14 smoke 的 0.9997 loss reduction
   是跨记录首末伪像;per-record 化后同一门在 64 updates 内三族全部不过。这不是回归,
   是首次得到未被伪像污染的数字。
2. **uniform 出现真实退化**(nonworse=false,achieved_gain -0.0068),且校正能量极小
   (ratio 2.9e-4)。confined 构造保证掩码外逐位等于 parent,因此退化发生在掩码内 —— r4e9
   oracle 证明掩码内存在 +0.124 的改��空间,但 64 步 SGD 不仅没拿到,还略微变差。
3. **门的可达性问题(记录,不裁决)**:0.80 的 per-record loss reduction 与 50% oracle 分数
   这两个阈值继承自 v14,而 v14 的"通过"证据现已知全部来自坏门。没有任何先验证据表明
   这两个阈值在 192-update smoke 预算内可达。按 veto (a) 阈值不得向实测值移动,本记录
   不修改任何阈值;可达性重估只能发生在未来候选(如 v16)的全新预注册里,且必须披露
   其设计参考了本次实测。
4. failure signal 记账:本次消耗一次 `failed_gate`。
5. smoke 证据不得晋级(veto f);本次为 fail,更无晋级问题。

## 遗留与下一步(须 lead 裁决,本记录不推进)

- 诊断方向 A:优化不足(64 步 vs oracle 的全窗最小二乘)。可用 train-only 探针
  (r4e8/r4e9 风格,非 stage)测同一 ansatz 在更多步数/不同 lr 下的收敛轨迹,零阈值改动。
- 诊断方向 B:uniform 掩码内退化的机制(掩码内保留 208/401 帧,保留帧初始损失已低至 0.0179,
  梯度信号弱;lr 0.003 对该尺度可能过大)。
- v15 按预注册在 fail_gate 下不得进入 pilot。任何 v16 预注册必须携带本记录与
  post_hoc_rule_revision 的完整历史。
