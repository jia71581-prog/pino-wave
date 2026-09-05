# r4e10 探针规格:v15 smoke fail_gate 的两支诊断(冻结于运行前)

日期:2026-08-26。类别:**train-only 诊断探针,非 stage,非晋级候选**。
两腿并行:leg A(GPU 0,优化收敛轨迹)与 leg B(GPU 1,uniform 掩码内退化机制)。

## 背景与问题

v15 smoke(`results/r16_dscp_v15/smoke/terminal.json`,status=fail_gate)在修好 v14 四项缺陷后
给出第一个真实读数:64 updates/记录 下 achieved (iii) gain 仅为本记录重算 oracle 上界的 0~25%,
uniform 为负(−0.0068,nonworse=false),校正能量比仅 2.9e-4。

两个互斥不完全的假设:

- **H-A(优化不足)**:同一 ansatz 在更多步数/更合适 lr 下能接近 oracle 上界;smoke 预算是绑定约束。
- **H-B(结构性瓶颈)**:1202 参数特征头的表达力/可训性,或 tanh·scale 参数化上限,
  使 SGD 不可达 oracle;加步数无用。

leg B 独立诊断 uniform 的净负机制(方向失配 / 梯度信号弱 / lr 过冲 / 参数化饱和)。

## 绑定(启动前必须逐一复核,漂移即拒)

| 文件 | sha256 |
|---|---|
| `saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py` | f906a422c26c86a736e21f77842ee83892ce7bd48364e2ca665a261451070201 |
| `saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py` | 959619b032a77af8f6095ba927e46fa30502df338f8067179ea4a94cfd4945a9 |
| `saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v3.py` | 8e0ae9294ff615633c07a1c686577325ed6785bf46ffb07d98e3e393963ba517 |
| `saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v15.py` | 4a3ca6fb5792347055cb1d7af869713e430ef9a2b59a7921b667fdd0915602ce |
| `results/r16_dscp_v1/basis_rank16.pt` | 8cab01344fc0a88f43fa232458127e87d2f8bbb63523b9389b927b39389c4786 |
| `results/r16_dscp_v1/panels.json` | 6d2f2facd86b449c058f9f824895f77977a23fbecf954c0de69144ec50162464 |
| `results/r16_dscp_v15/smoke/terminal.json`(只读参照) | 0947af12a248bb92e77f417a3dcd6b41dc922118afb68d76d94985ce15dccf35 |
| `results/r16_dscp_v15/smoke/last.pt`(只读,leg B 加载) | c8e54e0335f53185d45a6fb3c8f4b56de09d9936c49602ff7d58228628a9857c |
| 父检查点 A3 `epoch_0007.pt` | 448035bd0061205c67799eeef3b023a71b49e2db7028b6155078be1a886de789(前后各复算一次,漂移即停) |

## 共享管线(两腿一致)

- 记录:smoke 同三条 train 记录 train_uniform_00321 / train_layered_00564 / train_marmousi_00385。
  **只读 train split;validation/test_id 不可触。**
- 数据/特征路径复刻 v10 工厂 + v15 引擎:GuardedOnsetDataset → canonical_public →
  parent_predictor(dense_normalized, time_block=16)→ grid_eikonal_travel_time →
  deployment_args → deployment_features;模型 R16DSCP(冻结 basis+scales),
  configure_determinism(372);损失 = `masked_confined_loss`(training_v3,不改任何权重);
  校正路径 = v15 `_materialize` 等价(c1 ramp、首帧置零、parent 能量 keep mask 内 confined)。
  偏差声明:探针全程 fp32(smoke 的 `_coefficients` 用 bf16 autocast)。零初始化下初始
  系数恒为 0,初始损失不受此影响;训练轨迹可能有微小数值差,结论不依赖逐位一致。
- **管线保真门(先于一切测量)**:每记录 (1) parent_rel_l2 与 smoke terminal 相对差 ≤1e-3;
  (2) 零步初始损失与 smoke initial_loss 相对差 ≤1e-3;(3) 本探针重算 oracle 上界
  (同 `confined_oracle_upper_bound`)与 smoke terminal 相对差 ≤1e-6。任一不过 = 探针作废(invalid_pipeline),不得解读。

## leg A:收敛轨迹 / lr 扫描(GPU 0)

- 模型臂:lr ∈ {3e-4, 1e-3, 3e-3, 1e-2},每臂独立 seed 372 重新初始化,AdamW
  betas (0.9,0.99) eps 1e-8 wd 1e-4,clip 1.0(全部同 v15,仅 lr 变量);
  每臂 768 updates round-robin(=256/记录,4×smoke 预算)。每 48 updates 记录:
  每记录训练损失 + (iii) achieved gain。
- 直拟臂(参数化上界):每记录独立优化原始场 v[16,H,W](init 0),
  coefficients = tanh(v)·scale(与模型同参数化、同 confined 校正、同损失),
  Adam lr 3e-2,512 步。测"绕过 1202 参数头后,该参数化+该优化器可达什么"。
- 饱和度量:对每记录 oracle 逐像素系数图 â,报 frac(|â| > 0.99·scale)。

### 预注册判读(先于结果冻结)

以 g* = 该记录 oracle (iii) 上界,f(arm) = achieved/g*:
- (A1) 若最优 lr 臂在 ≥2/3 记录 f ≥ 0.5 → **H-A 成立**:预算是绑定约束;
  smoke 门在 192-update 预算下的不可达性得到实证(不构成调门理由,veto (a) 依旧)。
- (A2) 若所有模型臂在全部记录 f < 0.25,而直拟臂 f ≥ 0.5 → **H-B(特征头)成立**。
- (A3) 若直拟臂也 f < 0.5 且饱和分数 > 0.10 → **H-B(tanh·scale 参数化上限)成立**。
- (A4) 其余 → 按记录分支报告,不下单一结论。

## leg B:uniform 掩码内退化机制(GPU 1)

对 uniform(其余两族做对照,仅便宜量):
1. **方向**:加载 smoke `last.pt` 权重,materialize 校正 c,报掩码内
   cos(c, truth−parent)(全局 + 逐帧),及逐帧 (iii) 增益贡献。
2. **饱和**:oracle 系数图 vs scale 界(同 leg A 度量,uniform 重点)。
3. **梯度信号**:seed-372 初始处每记录 ‖∇θL‖(unclipped),三族对比。
4. **lr 过冲**:初始处对 uniform 各做一次单步 AdamW(lr ∈ {3e-5,3e-4,3e-3,3e-2},
   每次独立 fresh init),报 Δloss。

### 预注册判读

- (B1) smoke-final 掩码内 cos(c, e) < 0 → 方向失配实证。
- (B2) uniform ‖∇L‖ < 0.1 × 其余两族中位 → 弱梯度信号实证。
- (B3) 单步 Δloss(3e-3) > 0 且 Δloss(3e-4) < 0 → lr 过冲实证。
- (B4) uniform 饱和分数 > 0.10 → 参数化上限参与。
- 多条可同时成立;全部不成立则报 unexplained 并列出排除项。

## 预算与安全

- 每腿 GPU ≤ 1800 s,peak reserved ≤ 20 GiB(4090D 24G);超时/超限即中止并落 failed terminal。
- 磁盘:启动前 `/root/autodl-tmp` 空余 ≥ 2 GiB;产物 ≤ 64 MiB/腿(仅 JSON,零场数组序列化)。
- 写入白名单:`results/r4e10_optimization_gap_20260826/leg_{a,b}/` 新建文件;
  **零修改既有文件,零删除,零检查点写入**(leg A 的模型权重不落盘)。
- 父检查点与 v15 smoke 产物只读;运行前后各复算父 sha256 一次。
- 失败信号:invalid_pipeline / NaN-Inf / OOM / budget / binding_drift。

## 声明边界(继承全部在册 veto)

- 本探针产出**诊断证据**,不是性能达标证据;不得据此晋级任何 stage(veto f 类推)。
- 不得把任何阈值向实测值移动(veto a);A1 成立也只说明"该预算下不可达",
  阈值/预算修订只能发生在 v16 全新预注册并披露参考了本探针。
- 校正仍是输出场数据拟合 + 特征驱动 ansatz,零 PDE 残差(veto e)。
- oracle 是上界不是已达成结果;masked (ii) 只是训练代理,验收口径是 (iii)。
