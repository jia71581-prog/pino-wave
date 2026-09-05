# 磁盘清理记录(2026-08-27)

授权:lead 指令"检查点保留每种方法最后一个,中间步可以删除"(此前审计
提案经 lead"确认")。

## 已执行

- wkb*/direct*(conditioning/propagator prior 线,8-23 全线被拒)共 49 个
  run 目录:删除中间 update_*.pt **333 个,回收 34.8 GB**。
  每 run 保留:latest.pt、best.pt(如存在)、编号最末的 update_*.pt,
  合计 8.2 GB;terminal.json / preregistration / metrics.jsonl /
  updates.jsonl / 日志全部未动。证据链完整,末档 checkpoint 保续训能力。
- 磁盘:369G/380G(97%)→ 334G/380G(88%),可用 12G→47G。

## 提议但未执行(权限层拒绝,需 lead 以 ! 命令自行执行或明示指名)

1. `home/*/Data/FNO-Acoustic-Wave-Simulation/1/evaluation/marmousi_fixed19hz_position_240_post_r5d_r1/predictions/`
   ~14 GB 预测场 .npz 转储(score_summary/figures/manifest 保留,可由
   checkpoint 再生)。
2. `home/*/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_smoke/`
   649 MB 冒烟目录(正式变体 r1/r2_dense2/r3_adamw99 未动)。

## 未触碰(保留清单)

helmholtz*(25G,CLAUDE.md 明列)、a3_alltrain_*、cpadc、instance_adaptation、
baselines、Data/data/processed travel 缓存、r16_dscp 全链、A3/B2-H/ASAM 检查点。
