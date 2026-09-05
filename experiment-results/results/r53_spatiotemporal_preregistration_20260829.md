# R53 预注册:完整时空算子校正网络(2026-08-29,R52 结果揭晓前冻结)

## 动机(lead 指令"用完整的算子网络训练"的证据兼容落点)
- 纯完整算子历史全败(grouped/A+1 受限、factorized FNO P_bg val 0.99 rejected);R46 完整 FNO 校正器从零训恒等塌缩 → "完整算子网络"必须以热启动+identity 初始化进入。
- 现有校正器逐帧独立,无时间耦合;频散误差沿时间累积、强结构化;V72 定论"达标须结构改动(3D 时空)"。
- R53 = R39 2D 校正器(整记录 64 帧作 batch)+ 零初始化膨胀 3D 时间混合层(54,241 参,dilation 1/2/4,时间感受野 ±14 帧),加载 R39/R52 检查点后函数精确一致(CPU 核验 diff=0.0)。

## 配置
- script scripts/train_r53_spatiotemporal_tail.py;数据 r38_full_coverage_cache(记录级读取,num_workers=4 修复吞吐)。
- 记录级采样沿用 R29A 重复规则(1+marmousi+q75+q90,2128→3533 实例);损失 = 审计 R26 tail risk loss,CVaR 改为记录内(帧维);eval 复用 r25.evaluate(batch=64=整记录按序)。
- lr:base 1e-5(R52-A ep2 实证 max 0.0658→0.0645 下降背书)、HFS 恒 1e-4、temporal 1e-4(零起点新参数),cosine;8 epoch 全步数(~883 步/epoch),bf16 amp,seed 530829。
- **init 选择规则(现在冻结,R52 完成后执行)**:在 {r39v3 best, R52-A best, R52-B best} 中取 best.json score(=dev max+mean)最小者;R52-B 须先过其 identity 门。
- 长训前 1×20 smoke;identity 门:R53 initial_holdout 的 mean/max 与所选 init 检查点 best 指标各差 ≤ 2e-4,不过则废止。

## 判读(冻结)
- 改善信号:任一 epoch dev max ≤ min(0.0630, init_max − 0.002) 且该 epoch mean ≤ init_mean + 0.0005。
- 过冲失败:任一 epoch mean ≥ 0.0250。
- 无改善 → 定论"时间耦合(此实现)无增益",证据计入结构变体账本。
- 最终成功门不变:dev mean 且 max ≤ 0.05 双过才授权后续;本实验不作达标声明。

## 预算 / 回滚 / 边界
- 预算:smoke ~5min + 8 epoch ≈ 3h,单次不追加;VRAM 若 OOM 记录后停(24G 卡,64 帧/步)。
- 回滚:全部 init 检查点只读;独立输出目录。
- 数据边界:fit=训练,dev holdout(R28 已开)=评估;R29B/validation/test_id 不开。
