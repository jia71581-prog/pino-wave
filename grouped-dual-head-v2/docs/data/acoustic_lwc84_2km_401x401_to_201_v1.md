# Acoustic LWC-84 2 km：401×401 求解、201×201 保存、401 帧

## 冻结合同

- 配置：`configs/datasets/acoustic_lwc84_2km_401x401_to_201_v1.yaml`
- 物理节点：`401×401`，`x,z=0,5,…,2000 m`
- 保存节点：`201×201`，`x,z=0,10,…,2000 m`
- 降采样：可分离 binomial-5 低通后 2 倍 restriction；速度和波场均禁止直接 stride 抽样
- source_map：按连续震源坐标在 201 网格重新双线性离散，权重和为 1
- 时间：`0–1 s`，`dt_requested=0.0002 s`，冻结 `dt_used=0.000125 s`、stride=20、`dt_out=0.0025 s`、`nt_out=401`
- 轴顺序：波场 `NTZX=[sample,time,z,x]`

## HDF5 shard

每个 worker 只写自己的 shard。波场为 `[N,401,201,201] float32`，chunk `(1,1,201,201)`、LZF；速度和 source_map 为 `[N,201,201] float32`。根数据集还包含 source wavelet/坐标/频率/峰值时间/幅值、medium/sample/group/seed、CFL/LWC/QC/crop 元数据及坐标轴。

写入协议：

1. 写 `<shard>.h5.tmp`；
2. 每个样本完成后写 `completed_mask` 和样本 SHA-256，并 flush；
3. resume 时校验 schema、坐标、config hash、manifest hash，只续写未完成样本；
4. 全部完成后关闭并重开，逐样本严格检查 shape、dtype、有限性、source_map、QC 和 checksum；
5. 原子重命名为 `.h5`，再原子写 `.h5.sha256`；
6. VDS 只接纳具备正确 sidecar 的最终 `.h5`，永不接纳 `.tmp`。

全部 shard 完成后，启动器自动生成唯一用户入口 `dataset_v1.h5`。该文件是跨
`train/validation/test_id/ood_canonical` 的零拷贝 HDF5 VDS，包含逐样本 `split`、
`split_id` 和 `medium_type`；底层 shard 保留用于断点续跑和校验，不能移动或删除。

旧训练配置如果请求 `nu`/`tensor`，加载器可显式解析为 `velocity_mps`/`wavefield`；但新 schema 必须声明并遵守 `NTZX`，不会靠方形空间维度掩盖 `x/z` 轴交换。

## 数据组成

普通池共 4000：train/validation/ID-test=`2800/600/600`。介质合计：uniform 600、layered 1600、anomaly 800、用户授权插值派生 Marmousi crop 1000。另有 3 个独立 canonical OOD：`c=4000,f0=15`；`3000/5000 m/s` 两层、界面 1000 m、`f0=15`；Marmousi holdout、`f0=10`。

manifest 在波场之前冻结；group 不跨 split；Marmousi OOD crop 外扩 250 m guard band；归一化仅用 train shard 的 Welford 流式统计。

## 当前生产状态

只读审计确认原始 `/home/jiayh/pinn_fwi-main/data_model/marmousi_bl.bin` 是 `[116,227] float32`、1500–4450 m/s。按用户后续明确授权，系统使用归一化坐标线性插值生成 `[801,2401]` 派生模型，目标元数据为 5 m、`4000×12000 m`。派生文件 SHA-256 为 `212747a8ad08d95a8bf7cb0910971f130a407e8cd65b7f5f8b8ab5eb20f39aca`，原始文件和派生 provenance 均保留。该操作拉伸已有像素，不增加独立地质信息，因此所有配置和元数据均标记 `normalized_extent_stretch`，不称其为新的真实观测。

插值后 4000+3 manifest 已冻结：group leakage、频率窗口 leakage、Marmousi OOD/guard-band overlap 均为 0。OOD crop 为 `[5000,7000]×[0,2000] m`，guard band 为 `[4750,7250]×[0,2250] m`。Marmousi 尺寸门禁已解除。

401 帧下的容量与磁盘预算由 planner 每次根据冻结配置重新计算；临时空间只计并发写入的 partial shard，不重复计算整套最终数据。

## 已验证命令

```bash
python scripts/generate_acoustic_dataset.py --config configs/datasets/acoustic_lwc84_2km_401x401_to_201_v1.yaml --preset unit --device cpu --num-samples 4 --output artifacts/dataset_smoke_cpu_lwc84_final_schema
python scripts/generate_acoustic_dataset.py --config configs/datasets/acoustic_lwc84_2km_401x401_to_201_v1.yaml --preset smoke --device cuda --num-samples 1 --output artifacts/dataset_smoke_gpu_full401_lwc84_final_schema
python scripts/validate_acoustic_dataset.py --input artifacts/dataset_smoke_gpu_full401_lwc84_final_schema --strict
python scripts/analyze_lwc84_dispersion.py --output artifacts/lwc84_dispersion_401to201_801
python scripts/validate_lwc84_numerics.py --device cuda --output artifacts/lwc84_numerical_validation_401to201_801
```

## 正式入口

```bash
python scripts/launch_gpu_dataset_workers.py --config configs/datasets/acoustic_lwc84_2km_401x401_to_201_v1.yaml --output /home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1 --devices 0 --batch-size 128 --resume --confirm-production RUN_4000
```

生产已通过 GPU、磁盘、Marmousi、manifest 和泄漏门禁。运行时 batch=128 跨 16 个
8-sample shard 联合正演，再分别原子写回 shard；最终自动建立单一 `dataset_v1.h5`。
