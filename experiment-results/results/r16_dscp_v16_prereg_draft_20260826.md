# r16_dscp v16 预注册草案(待 lead 确认后冻结)

日期:2026-08-26。依据:r4e10(sha 见其 spec)→ r4e11(C4)→ r4e12(**D1+W+**)三级探针证据链。
本文件是**草案**,不是冻结预注册;冻结前须 lead 确认三处标注 [LEAD] 的裁量项。

## 候选定义

- 名称:`r16_dscp_v16_wide128`。
- 结构改动(相对 v15 唯一改动):1202 参 coefficient_head → M2 拓扑宽 128
  (depthwise-separable 29→128→128→16,3×3/dilation1,SiLU,末层零初始化,25266 参,
  r4e12 W 臂逐字)。tanh × 冻结 per-family scale 输出契约、confined 校正路径、
  parent 能量掩码、29 通道特征、basis_rank16、TAU 全部不变。
- 训练:field 单目标(masked_confined_loss),AdamW(0.9,0.99)/eps1e-8/wd1e-4/clip1.0,
  lr 3e-3 恒定(r4e12:cosine 无收益),seed 372。
- 预算:smoke = **3072 updates**(v15 的 192 已被 r4e10/r4e12 证实远低于收敛地平线;
  W 臂 3072 时 2/3 记录过 0.5 且仍爬)。

## 门(阈值重估的证据披露,veto (a) 合规声明)

v15 的 0.80 loss-reduction / 50% oracle-gain 阈值继承自 v14 坏门时代,v15 smoke 实测
证明其在 192-update 预算不可达;本草案按 r4e12 实测轨迹**在新预注册中先验设定**
(不是用结果调已冻结门;v15 门保持原样,其 fail_gate 记账不变):

- smoke loss 门(per-record,v15 修正口径):[LEAD] 建议阈值 0.30(r4e12 L 臂 3072
  实测 loss reduction 下界的保守折扣;W 臂更高)。
- oracle_gain 门(口径 iii,本记录重算上界):achieved ≥ 上界的 [LEAD] 建议 35%
  于 ≥2/3 记录(r4e12 W = 44%/68%/81%,留泛化损耗余量;uniform 44% 是三族最低)。
- nonworse 门:3/3 记录 aggregate_rel_l2 不劣于 parent(r4e12 W 三记录全改善,
  该门从 2/3 收紧为 3/3)。[LEAD]
- vram/时延/checkpoint ���沿用 v15(25k 参 fp32 100 KB,前向亚毫秒,均有巨量余量)。

## 晋级路径与记账

smoke(3 train 记录)→ pilot(train 全集 + validation 首次开封)→ long。
smoke 只证收敛与实现正确,**泛化证据从 pilot 起算**;v15 的 1 次 failed_gate 记账
保留,v16 是新候选新记账。探针 r4e10-12 全部引用为设计依据并附 sha。

## 已知风险(如实声明)

1. 3 记录 train-only → 泛化未证;探针全程 round-robin 同 3 记录,pilot 需全 train 集
   多记录训练,收敛地平线可能右移。
2. uniform 族 f=0.44:其 oracle 上界 0.124 为三族最小,rank16 表征对 uniform 晚期
   逸出场的天花板(r4e8:r32−r16 增量 +0.135)未在本线解决;v16 不改 rank(留量另案)。
3. marmousi parent 0.60 太差,校正后仍 0.54:头修不了 basis 天花板,预期管理写入 prereg。
4. lr 3e-3 在 1202 参头上曾致 uniform 过冲(r4e10 B3);25k 头 r4e12 未复现,
   但 pilot 多记录下须保留发散哨兵(损失连升 3 评估点即停)。

## 绑定清单(冻结时逐项 sha)

r4e10 的 9 项 + probe r4e11/r4e12 脚本与 terminal + 本草案定稿 sha。
