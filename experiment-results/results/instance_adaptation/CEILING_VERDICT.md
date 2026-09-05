# 信息含量天花板研究 —— VERDICT

日期: 2026-07-28
承接: [[fno-acoustic-instance-adaptation]] REPORT "后续可选 #1"(放宽 K 到含中/晚期稀疏帧)与
[[fno-acoustic-v72-experiment]] 的结构地板定论。

## 研究问题
既定结论是"held-out relL2 卡在 0.47–0.51 是**架构/容量结构地板**"。本研究要回答一个正交且更根本的问题:
**这道墙到底是"模型容量"限制,还是"观测信息含量"限制?** 即——如果给一个**容量无上限**的校正器
(自由全场张量,零结构约束),再喂给它不同数量的真实波场帧,held-out 误差能压到多低?

两个探针,共同夹逼答案:

1. **`diagnose_sparse_frame_ceiling.py`(信息含量上界,非因果)**: 每实例校正 = 一个与父场同形的
   **自由全场张量**(`correction = zeros_like(parent); parent + correction`),Adam 直接优化,**无任何容量瓶颈**。
   观测 = K 帧沿能量帧区间均匀铺开(early+mid+late);held-out = 其余能量帧(能量 > 峰值 5%)。
   损失 = 整场能量归一观测项(**不是** V72/REPORT 里那个对近零早期帧发散的逐帧比值)+ LWC-84 自归一 PDE。
   这是**理论可行性上界**:允许观测任意时刻,衡量"信息够不够",与因果性无关。

2. **`diagnose_highcap_instance_finetune.py`(因果 2 帧契约天花板)**: 严守因果契约——**只读 2 早期起振帧**,
   解冻整条残差头(13.4k 参数)+ PDE 自监督推未观测时刻。修正了原脚本的观测损失 bug
   (逐近零帧比值 → obs_loss≈2777 把 held-out 推到 0.91;改为整场能量归一,与 sparse 脚本一致)。

## 结果

### 探针 1 —— 自由全场校正的信息含量天花板(best held-out relL2)

| family | 父基线 | K=6 | K=12 | K=24 | K=48 | K=100 |
|--------|-------|-----|------|------|------|-------|
| uniform  | 0.506 | 0.448 | 0.382 | 0.300 | **0.139** | **0.122** |
| layered  | 0.214 | 0.197 | 0.179 | 0.123 | **0.072** | **0.061** |
| marmousi | 0.520 | 0.407 | 0.327 | 0.275 | **0.135** | **0.117** |

观测帧本身恒被拟合到 ~0.003–0.01(容量确实无限)。K=401 中取 ~48 帧即可让**零结构**校正器
把三族 held-out 全压到 **0.06–0.14**,逼近 capacity_ladder 过拟合地板(0.125–0.18)乃至 0.10 目标。

### 探针 2 —— 因果 2 帧契约(修正 obs-loss 后)

| family | 父基线 | 2 帧 best held-out |
|--------|-------|-------------------|
| uniform  | 0.507 | 0.535(未超越) |
| layered  | 0.217 | 0.217(=父,门控回退) |
| marmousi | 0.521 | 0.522(未超越) |

修正观测损失 bug 后不再发散(原为 0.91),但 2 早期帧 + PDE **仍无法超越冻结父场**。

## 判定

**这道墙的性质被重新定性:不是纯"模型容量地板",而是"观测信息含量地板"。**

- **单调、强烈依赖观测密度**: 同一个零结构自由校正器,held-out 随观测帧数 6→100 从 ~0.45 掉到 ~0.12。
  容量恒为无限却随信息量剧变 → **限制因素是信息,不是容量**。这与 capacity_ladder"加参数/加宽无用"完全自洽:
  加模型容量不给新信息,自然无效;加观测帧给的是新信息,立刻见效。
- **因果契约是真正的紧约束**: 2 早期帧(契约允许的全部)信息量太少,连无限容量都超不过父场(探针 2)。
  [[fno-acoustic-instance-adaptation]] 的 33 参部署 LoRA "改善极小"由此获得根因解释——**不是微调层太小,
  是 2 帧观测的信息含量本就不足以重建晚期/均匀锐波前**。
- **可达性坐标**: 若放宽因果契约到"稀疏采样 ~48+ 真实时刻"(如实测场景可陆续获得后续快照的在线同化),
  0.10–0.14 是可达的,且**不需要任何结构改动**——现成自由校正即可。反之在严格 early-only 契约下,
  唯一出路是 memory option B(改 coarse MIONet 结构,给模型更强的物理先验去外推它没观测到的时刻)。

**一句话**: 0.47 墙 = 信息墙 ∩ 结构墙。加容量撞结构墙(已证);加观测帧穿透信息墙(本证);
严格 2 帧因果下两墙叠加,无捷径,只能靠结构先验(option B)或放宽观测契约。

## 产物
- `results/instance_adaptation/ceiling_study_summary.json`(机读汇总)
- `results/instance_adaptation/ceiling_sweep/*.json`(15 次 sparse 扫描原始输出)
- `results/instance_adaptation/highcap_2frame/*.json`(3 族因果 2 帧)
- 脚本: `scripts/diagnose_sparse_frame_ceiling.py`, `scripts/diagnose_highcap_instance_finetune.py`(obs-loss 已修),
  `scripts/run_sparse_ceiling_sweep.sh`, `scripts/run_highcap_2frame.sh`

## 复现
```bash
WORKROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2; cd $WORKROOT
bash scripts/run_sparse_ceiling_sweep.sh 0 validation_uniform_00069 validation_layered_00048
bash scripts/run_sparse_ceiling_sweep.sh 3 validation_marmousi_00109
bash scripts/run_highcap_2frame.sh 3 validation_uniform_00069 validation_layered_00048 validation_marmousi_00109
```
