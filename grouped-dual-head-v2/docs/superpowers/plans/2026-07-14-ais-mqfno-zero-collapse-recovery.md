# AIS-MQFNO Zero-Collapse Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed raw-scale AIS screen with normalization-contract-v2 training and select among eight independently trained MQFNO variants using category-aware successive halving without opening test data.

**Architecture:** A focused normalization module owns train-only statistics, dimensionless features, physical decoding, and checkpoint binding. Optional multiscale-local and dispersion-residual components plug into the existing full160 query model through explicit configuration fields. A separate pure screen module consumes immutable validation artifacts and makes deterministic Gate O/H1/H2/H3 decisions.

**Tech Stack:** Python 3.10, PyTorch, HDF5/h5py, NumPy, PyYAML, pytest, ruff, CUDA on RTX 3090.

---

## File map

- `src/fno_acoustic/ais_normalization.py`: normalization loading, hashing, feature encoding, physical wavefield conversion.
- `src/fno_acoustic/query_losses.py`: standardized MSE plus physical HH relative loss.
- `src/fno_acoustic/query_training.py`: normalized training/validation and schema-v4 checkpoint binding.
- `src/fno_acoustic/ais_model_components.py`: multiscale local encoder and dispersion residual head.
- `src/fno_acoustic/model_ais_mqfno.py`: configured ordinary/multiscale/dispersion paths.
- `src/fno_acoustic/ais_screen.py`: gate records, exclusions, scores, and promotions.
- `scripts/generate_ais_v2_configs.py`: exactly eight materialized configurations.
- `scripts/run_ais_v2_screen.py`: serial orchestration and immutable manifest.
- `scripts/train_ais_mqfno.py`, `scripts/evaluate_ais_mqfno.py`: normalization-bound CLIs and pinned legacy baseline.
- `tests/test_ais_normalization.py`, `tests/test_standardized_hh_loss.py`, `tests/test_ais_checkpoint_normalization.py`, `tests/test_ais_model_variants.py`, `tests/test_ais_v2_configs.py`, `tests/test_ais_category_metrics.py`, `tests/test_ais_screen.py`, `tests/test_ais_v2_cli.py`: focused TDD coverage.

## Task 1: Normalization-contract-v2 primitives

**Files:**
- Create: `src/fno_acoustic/ais_normalization.py`
- Create: `tests/test_ais_normalization.py`
- Modify: `src/fno_acoustic/query_training.py:874-920`

- [ ] **Step 1: Write failing round-trip and hash tests**

```python
def test_wavefield_standardization_round_trips_physical_values(tmp_path):
    binding = make_binding(tmp_path, velocity=(3000.0, 500.0), wavefield=(0.1, 0.25))
    physical = torch.tensor([-0.4, 0.1, 0.6])
    encoded = binding.encode_wavefield(physical)
    assert torch.allclose(encoded, torch.tensor([-2.0, 0.0, 2.0]))
    assert torch.allclose(binding.decode_wavefield(encoded), physical)


def test_binding_hashes_exact_stats_bytes(tmp_path):
    binding = make_binding(tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2))
    assert binding.contract_id == "ais_normalization_v2"
    assert binding.stats_sha256 == hashlib.sha256(binding.path.read_bytes()).hexdigest()
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_normalization.py -q
```

Expected: FAIL because `fno_acoustic.ais_normalization` does not exist.

- [ ] **Step 3: Implement the immutable binding**

```python
@dataclass(frozen=True)
class AISNormalizationBinding:
    path: Path
    stats_sha256: str
    velocity_mean: float
    velocity_std: float
    wavefield_mean: float
    wavefield_std: float
    eps: float
    contract_id: str = "ais_normalization_v2"

    def encode_wavefield(self, value):
        return (value - self.wavefield_mean) / max(self.wavefield_std, self.eps)

    def decode_wavefield(self, value):
        return value * max(self.wavefield_std, self.eps) + self.wavefield_mean
```

`load_ais_normalization(path)` validates train provenance, finite means, positive standard deviations/eps, and hashes exact bytes. Reject booleans and nonfinite values.

- [ ] **Step 4: Write the failing dimensionless-feature test**

```python
def test_static_features_are_dimensionless_and_scale_balanced(tmp_path):
    binding = make_binding(tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2))
    scene = make_scene(height=5, width=7)
    features = build_normalized_static_features(scene, binding)
    assert features.shape == (1, 5, 5, 7)
    assert torch.allclose(features[:, 0], (scene.velocity_cpu - 3000.0) / 500.0)
    assert float(features[:, 1].abs().max()) == pytest.approx(1.0)
    assert torch.isfinite(features).all()
    assert float(features.abs().max()) < 20.0
```

- [ ] **Step 5: Run RED, then implement normalized static channels**

```python
def build_normalized_static_features(scene, binding):
    velocity_hat = (scene.velocity_cpu - binding.velocity_mean) / max(binding.velocity_std, binding.eps)
    source = scene.source_cpu / scene.source_cpu.abs().max().clamp_min(binding.eps)
    xi = torch.linspace(-1.0, 1.0, scene.velocity_cpu.shape[0])
    zeta = torch.linspace(-1.0, 1.0, scene.velocity_cpu.shape[1])
    grad_x, grad_z = torch.gradient(velocity_hat, spacing=(xi, zeta), edge_order=2)
    slow_contrast = (binding.velocity_mean / scene.velocity_cpu.clamp_min(binding.eps)).square() - 1.0
    return torch.stack((velocity_hat, source, grad_x, grad_z, slow_contrast))[None]
```

Change `build_scene_model_inputs(scene, global_size, device, normalization=None)`. `None` remains available only to pinned legacy evaluation and small historical test fixtures; formal v2 callers pass a binding.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_normalization.py tests/test_ais_query_training.py -q
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m ruff check src/fno_acoustic/ais_normalization.py src/fno_acoustic/query_training.py tests/test_ais_normalization.py
git add src/fno_acoustic/ais_normalization.py src/fno_acoustic/query_training.py tests/test_ais_normalization.py
git commit -m "feat: add AIS normalization v2 features"
```

## Task 2: Standardized HH loss and physical residual paths

**Files:**
- Modify: `src/fno_acoustic/query_losses.py`
- Modify: `src/fno_acoustic/query_training.py:1120-1540`
- Create: `tests/test_standardized_hh_loss.py`

- [ ] **Step 1: Write the failing loss-value test**

```python
def test_standardized_hh_uses_normalized_mse_and_physical_relative_l2():
    target = torch.tensor([[[0.0] * 160, [1.0] * 160]])
    prediction = target + 0.2
    binding = fake_binding(wavefield_mean=0.1, wavefield_std=0.5)
    target_hat, prediction_hat = binding.encode_wavefield(target), binding.encode_wavefield(prediction)
    result = standardized_hh_field_loss(prediction_hat, target_hat, torch.full((1, 2), .5), 2, binding)
    assert torch.allclose(result.mse, (prediction_hat - target_hat).square().mean())
    assert torch.allclose(result.relative_l2,
        torch.linalg.vector_norm(prediction-target) / torch.linalg.vector_norm(target))
```

- [ ] **Step 2: Run RED**

Expected: FAIL because `standardized_hh_field_loss` is missing.

- [ ] **Step 3: Implement the composed loss**

```python
def standardized_hh_field_loss(prediction_hat, target_hat, draw_probability_bq,
                               population_size, normalization,
                               late_time_weights=None, eps=1e-8):
    normalized = hansen_hurwitz_field_loss(
        prediction_hat, target_hat, draw_probability_bq, population_size,
        late_time_weights=late_time_weights, eps=eps)
    physical = hansen_hurwitz_field_loss(
        normalization.decode_wavefield(prediction_hat),
        normalization.decode_wavefield(target_hat), draw_probability_bq,
        population_size, late_time_weights=late_time_weights, eps=eps)
    return HHFieldLoss(normalized.mse + physical.relative_l2,
                       normalized.mse, physical.relative_l2)
```

- [ ] **Step 4: Write RED tests for physical sampler residuals and validation**

```python
def test_train_epoch_updates_sampler_with_decoded_physical_residual():
    scene, binding = make_scene(), fake_binding(wavefield_mean=.2, wavefield_std=.5)
    sampler = RecordingSampler()
    model = FixedEncodedPredictionModel(torch.full((1, 3, 160), .4))
    train_one_scene(model, sampler, scene, binding, torch.tensor([0, 1, 2]))
    physical = binding.decode_wavefield(torch.full((1, 3, 160), .4))
    target = scene.target_cpu.reshape(-1,160)[:3][None]
    assert torch.allclose(sampler.last_residual, (physical-target).square().mean(-1))

def test_validation_decodes_prediction_before_relative_l2():
    scene, binding = make_scene(), fake_binding(wavefield_mean=.2, wavefield_std=.5)
    indices = torch.tensor([0, 2, 4])
    exact_hat = binding.encode_wavefield(scene.target_cpu.reshape(-1,160)[indices][None])
    metrics = validate_fixed_prediction(scene, indices, exact_hat, binding)
    assert metrics['relative_l2'] == pytest.approx(0.0, abs=1e-7)
```

The first test uses mean `0.2`, std `0.5` and asserts the sampler receives `(decode(pred_hat)-target_physical)**2`. The second uses an exact encoded-target model and expects physical relative L2 zero.

- [ ] **Step 5: Route encoded labels through bounded backward**

Pass the binding through `train_query_epoch`, `_memory_bounded_field_backward`, and `validate_query_guard`. Encode only gathered labels; decode predictions before auxiliary losses, residual updates, and metrics. Never move `[400,400,160]` labels to GPU.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_standardized_hh_loss.py tests/test_hansen_hurwitz_loss.py tests/test_ais_query_training.py -q
git add src/fno_acoustic/query_losses.py src/fno_acoustic/query_training.py tests/test_standardized_hh_loss.py
git commit -m "feat: train AIS in normalized wavefield space"
```

## Task 3: Schema-v4 normalization binding

**Files:**
- Modify: `src/fno_acoustic/query_training.py:260-840`
- Modify: `scripts/train_ais_mqfno.py`
- Modify: `scripts/evaluate_ais_mqfno.py`
- Create: `tests/test_ais_checkpoint_normalization.py`

- [ ] **Step 1: Write RED checkpoint tests**

```python
def test_schema_v4_checkpoint_binds_normalization_hash(tmp_path):
    save_minimal_checkpoint(tmp_path/'last.pt', schema_version=4,
        normalization_stats_sha256='a'*64,
        normalization_contract='ais_normalization_v2')
    payload = torch.load(tmp_path/'last.pt', weights_only=True)
    assert payload['schema_version'] == 4

def test_v2_resume_rejects_v3_or_changed_stats(tmp_path):
    with pytest.raises(ValueError, match='normalization'):
        load_v2_checkpoint(checkpoint_v3(tmp_path), expected_normalization_sha256='a'*64)
```

- [ ] **Step 2: Run RED**

Expected: unsupported schema or extra-field failure.

- [ ] **Step 3: Add schema-v4 fields and pre-mutation validation**

```python
QUERY_CHECKPOINT_V4_FIELDS = QUERY_CHECKPOINT_V3_FIELDS | {
    'normalization_stats_sha256', 'normalization_contract'
}
```

Formal v2 callers require v4. Generic historical tests may read v1--v3 only when no expected normalization binding is supplied. Validate hashes before mutating model, optimizer, scheduler, sampler, or RNG.

- [ ] **Step 4: Add a failing CLI mutation test, then bind both CLIs**

```python
def test_train_cli_writes_v4_and_rejects_stats_change(tmp_path):
    run_cli_once(tmp_path)
    assert torch.load(tmp_path/'run/checkpoints/last.pt', weights_only=True)['schema_version'] == 4
    mutate_stats_bytes(tmp_path)
    resumed = run_cli_resume(tmp_path)
    assert resumed.returncode != 0 and 'normalization' in resumed.stderr
```

Training loads one binding and stores it in every checkpoint. Evaluation validates the same hash before model construction.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_checkpoint_normalization.py tests/test_ais_checkpoint_resume.py tests/test_ais_query_training.py -q
git add src/fno_acoustic/query_training.py scripts/train_ais_mqfno.py scripts/evaluate_ais_mqfno.py tests/test_ais_checkpoint_normalization.py
git commit -m "feat: bind AIS checkpoints to normalization v2"
```

## Task 4: Independent multiscale-local encoder

**Files:**
- Create: `src/fno_acoustic/ais_model_components.py`
- Modify: `src/fno_acoustic/model_ais_mqfno.py`
- Create: `tests/test_ais_model_variants.py`

- [ ] **Step 1: Write RED structure and gradient tests**

```python
def test_multiscale_local_uses_independent_patch9_and_patch25_parameters():
    encoder = MultiScaleLocalQueryEncoder(5, branch_dim=12)
    assert encoder.output_dim == 24
    assert encoder.small.halo_size == 9 and encoder.large.halo_size == 25
    assert not any(a.data_ptr() == b.data_ptr()
                   for a in encoder.small.parameters()
                   for b in encoder.large.parameters())

def test_multiscale_local_backpropagates_to_both_branches():
    encoder(native_static, query_xz).square().mean().backward()
    assert all(p.grad is not None for p in encoder.small.parameters())
    assert all(p.grad is not None for p in encoder.large.parameters())
```

- [ ] **Step 2: Run RED**

Expected: import failure for `ais_model_components`.

- [ ] **Step 3: Implement the exact two-branch component**

```python
class MultiScaleLocalQueryEncoder(nn.Module):
    output_dim = 24
    def __init__(self, in_channels, branch_dim=12):
        super().__init__()
        if branch_dim != 12:
            raise ValueError('normalization-v2 screen fixes branch_dim=12')
        self.small = LocalQueryEncoder(in_channels, branch_dim, 9)
        self.large = LocalQueryEncoder(in_channels, branch_dim, 25)
    def forward(self, native_static, query_xz):
        return torch.cat((self.small(native_static, query_xz),
                          self.large(native_static, query_xz)), dim=-1)
```

- [ ] **Step 4: Add `local_encoder_kind` to AISMQFNO**

Accept only `single` and `multiscale_9_25`. Multiscale requires `local_dim=24`, `fusion_dim=48`, and uses the new component. Single preserves current behavior.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_model_variants.py tests/test_ais_mqfno.py -q
git add src/fno_acoustic/ais_model_components.py src/fno_acoustic/model_ais_mqfno.py tests/test_ais_model_variants.py
git commit -m "feat: add multiscale AIS local encoder"
```

## Task 5: Zero-initialized dispersion residual head

**Files:**
- Modify: `src/fno_acoustic/ais_model_components.py`
- Modify: `src/fno_acoustic/model_ais_mqfno.py`
- Modify: `tests/test_ais_model_variants.py`

- [ ] **Step 1: Write RED zero-init and basis-frequency tests**

```python
def test_dispersion_head_is_exactly_zero_at_initialization():
    head = DispersionResidualHead(16, 48, 24, 3000.0, 500.0)
    residual = head(local, velocity_hat, 30.0, 30.0, time_s)
    assert residual.shape == (2, 11, 160)
    assert torch.count_nonzero(residual) == 0

def test_dispersion_head_uses_basis_frequency_not_source_frequency():
    f = head.dispersion_features(torch.zeros(1,1), 30.0, 40.0, 0.5)
    assert f[0,0,0,1] == pytest.approx((1/0.5)*30/3000)
    assert f[0,0,23,1] == pytest.approx((24/0.5)*30/3000)
```

- [ ] **Step 2: Run RED**

Expected: `DispersionResidualHead` is missing.

- [ ] **Step 3: Implement the 24-mode real sin/cos head**

```python
class DispersionResidualHead(nn.Module):
    def __init__(self, local_dim, hidden_dim=48, modes=24,
                 velocity_mean=3000.0, velocity_std=500.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(local_dim+3, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
```

In `forward`, recover `v=mu+sigma*v_hat`; construct per `(query,m)` features `[local, m/24, (m/T)dx/v, (m/T)dz/v]`; emit real `(a,b)`; return `sum_m(a*cos(2πm(t-t0)/T)+b*sin(2πm(t-t0)/T))/sqrt(24)`.

- [ ] **Step 4: Integrate `dispersion_head=none|phase_residual_24`**

Add the residual to standardized trajectory output before physical decoding. Store velocity mean/std as immutable buffers. Extend `SceneModelInputs` with `dx_m,dz_m` and `decode_queries` with optional `dx_m,dz_m`; dispersion models require finite positive values, ordinary models accept `None`. Training/reconstruction pass scene spacings. The training CLI injects binding velocity mean/std into N6/N7 model kwargs; they are not independently configurable.

- [ ] **Step 5: Test two-step gradient activation**

At initialization the first backward must update the zero final projection. After one optimizer step, a second backward must give nonzero gradients to the preceding MLP and local encoder.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_model_variants.py tests/test_ais_mqfno.py -q
git add src/fno_acoustic/ais_model_components.py src/fno_acoustic/model_ais_mqfno.py tests/test_ais_model_variants.py
git commit -m "feat: add AIS dispersion residual head"
```

## Task 6: Generate exactly eight candidate configs

**Files:**
- Create: `scripts/generate_ais_v2_configs.py`
- Create: `tests/test_ais_v2_configs.py`
- Create: `configs/ais_zero_collapse_v2/n0_norm.yaml`
- Create: `configs/ais_zero_collapse_v2/n1_wide.yaml`
- Create: `configs/ais_zero_collapse_v2/n2_spatial.yaml`
- Create: `configs/ais_zero_collapse_v2/n3_temporal.yaml`
- Create: `configs/ais_zero_collapse_v2/n4_large_local.yaml`
- Create: `configs/ais_zero_collapse_v2/n5_multi_local.yaml`
- Create: `configs/ais_zero_collapse_v2/n6_dispersion.yaml`
- Create: `configs/ais_zero_collapse_v2/n7_multi_dispersion.yaml`

- [ ] **Step 1: Write RED inventory test**

```python
EXPECTED = {f'n{i}_'+suffix for i, suffix in enumerate(
 ['norm.yaml','wide.yaml','spatial.yaml','temporal.yaml','large_local.yaml',
  'multi_local.yaml','dispersion.yaml','multi_dispersion.yaml'])}
def test_generator_materializes_exactly_eight_candidates():
    configs = build_v2_configs()
    assert set(configs) == EXPECTED
    assert all(c['normalization']['contract'] == 'ais_normalization_v2'
               for c in configs.values())
```

- [ ] **Step 2: Run RED**

Expected: import failure for `scripts.generate_ais_v2_configs`.

- [ ] **Step 3: Implement exact overrides**

```python
VARIANTS = {
 'n0_norm.yaml': {},
 'n1_wide.yaml': {'spatial_width':32,'local_dim':24,'fusion_dim':48},
 'n2_spatial.yaml': {'spatial_width':32,'spatial_modes':32},
 'n3_temporal.yaml': {'temporal_modes':48,'fusion_dim':48},
 'n4_large_local.yaml': {'local_patch_size':25,'local_dim':24},
 'n5_multi_local.yaml': {'local_encoder_kind':'multiscale_9_25','local_dim':24,'fusion_dim':48},
 'n6_dispersion.yaml': {'dispersion_head':'phase_residual_24'},
 'n7_multi_dispersion.yaml': {'local_encoder_kind':'multiscale_9_25','local_dim':24,
                              'fusion_dim':48,'dispersion_head':'phase_residual_24'},
}
```

- [ ] **Step 4: Test equal budgets and unique structures**

Assert identical seed, data/split/stats, 2048 queries, full160, optimizer, scheduler, validation sites, and scene order. Assert Gate O/H1/H2/H3 totals are 400/600/1500/3000 and all canonical model dictionaries differ. Each config registers one common `screen.gate_o_sample_id` from train and one fixed 2048-site manifest SHA. The model phase limit stays 3000: Gate O uses 400 updates in a separate output, H1 restarts to 600, H2 resumes for 900 more, and H3 resumes for 1500 more.

- [ ] **Step 5: Generate, run GREEN, and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_v2_configs.py tests/test_ais_configs.py -q
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python scripts/generate_ais_v2_configs.py
git add scripts/generate_ais_v2_configs.py tests/test_ais_v2_configs.py configs/ais_zero_collapse_v2
git commit -m "feat: register eight AIS v2 candidates"
```

## Task 7: Category physical metrics and pinned legacy B1

**Files:**
- Modify: `src/fno_acoustic/query_census.py`
- Modify: `src/fno_acoustic/long_horizon_metrics.py`
- Modify: `scripts/evaluate_ais_mqfno.py`
- Create: `tests/test_ais_category_metrics.py`

- [ ] **Step 1: Write RED category tests**

```python
def test_category_report_records_physical_zero_baselines():
    report = aggregate_category_metrics(sample_rows())
    for category in ('uniform','layered','marmousi'):
        assert report[category]['zero_relative_l2'] == pytest.approx(1.0)
        assert report[category]['zero_relative_l2_q4'] == pytest.approx(1.0)
        assert 'prediction_target_norm_ratio' in report[category]
        assert 'prediction_target_pearson' in report[category]
```

- [ ] **Step 2: Run RED, then implement sample-first aggregation**

Compute metrics per physical scene and average within category. Never concatenate categories before ratios. Register the exact lower/higher metric names from the design plus finite/shape validity fields. `QueryPredictorAdapter` receives `AISNormalizationBinding` and decodes the complete reconstructed standardized field before census metrics.

- [ ] **Step 3: Write RED pinned-legacy tests**

```python
def test_legacy_b1_accepts_only_pinned_hash(tmp_path):
    assert load_legacy_raw_b1(PINNED_B1_PATH).sha256 == PINNED_B1_SHA256
    with pytest.raises(ValueError, match='pinned legacy B1'):
        load_legacy_raw_b1(tmp_path/'different.pt')

def test_legacy_b1_adapter_exposes_no_training_state():
    assert not hasattr(load_legacy_raw_b1(PINNED_B1_PATH), 'optimizer_state_dict')
```

- [ ] **Step 4: Add narrow `--legacy-raw-b1-baseline` evaluation mode**

Require validation split, exact checkpoint SHA `91546ba21c31e0e875c0ed7051068c279b455a94f38206274094e1a9a9233653`, original config hash, and raw semantics. Reject test, resume, initialization, representative tuning, recipe freeze, authorization, or sealed-runner use.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_category_metrics.py tests/test_long_horizon_metrics.py tests/test_query_census.py -q
git add src/fno_acoustic/query_census.py src/fno_acoustic/long_horizon_metrics.py scripts/evaluate_ais_mqfno.py tests/test_ais_category_metrics.py
git commit -m "feat: add category-aware AIS validation"
```

## Task 8: Deterministic Gate O/H1/H2/H3 decisions

**Files:**
- Create: `src/fno_acoustic/ais_screen.py`
- Create: `tests/test_ais_screen.py`

- [ ] **Step 1: Write RED Gate O tests**

```python
def test_gate_o_requires_n0_and_at_least_four_passes():
    decision = gate_o(records_with_passes('N0','N1','N2'))
    assert decision.status == 'stop'
    assert decision.reason == 'fewer_than_four_representable_candidates'

def test_gate_o_uses_all_fixed_thresholds():
    r = overfit_record(relative_l2=.35, q4=.50, norm_ratio=.5, pearson=.80, finite=True)
    assert gate_o_candidate(r).passed
```

- [ ] **Step 2: Run RED**

Expected: import failure for `fno_acoustic.ais_screen`.

- [ ] **Step 3: Implement frozen records and Gate O**

```python
@dataclass(frozen=True)
class CandidateGateRecord:
    candidate_id: str
    update: int
    checkpoint_sha256: str
    category_metrics: Mapping[str, Mapping[str, float]]
    finite: bool
```

Gate O enforces N0, at least four passes, relative `.35`, Q4 `.50`, norm `[.5,1.5]`, Pearson `.80`, and finite state without threshold arguments.

- [ ] **Step 4: Write RED H1/H2/final tests**

Cover per-category norm bounds; reporter zero fields; exclusions before ranking; exact updates 600/1500/3000; fewer-than-four/two stops; deterministic candidate-ID ties; per-category 30% field/Q4; exact 5% lower/higher guards.

```python
def test_h1_ranks_update600_last_after_exclusions():
    result = gate_h1(records_at_update_600())
    assert result.advanced == ('N7','N5','N3','N1')
```

- [ ] **Step 5: Implement score and promotion formulas**

```python
def ranking_score(categories):
    return max(v['relative_l2'] for v in categories.values()) + .5 * max(
        v['relative_l2_q4'] for v in categories.values())

def lower_error_guard(candidate, baseline):
    return candidate <= baseline + max(.05 * baseline, 1e-6)

LOWER_GUARDS = (
 'receiver_relative_l2','receiver_relative_l2_q1','receiver_relative_l2_q2',
 'receiver_relative_l2_q3','receiver_relative_l2_q4','arrival_mae_s',
 'arrival_miss_rate','receiver_lag_abs_s','receiver_phase_error',
 'energy_log_ratio','komega_relative_l2','komega_high',
 'komega_relative_l2_q4','komega_high_q4')
HIGHER_GUARDS = ('receiver_xcorr_peak','receiver_phase_coherence')
```

For every category, require final `relative_l2` and `relative_l2_q4` at most `.70 * baseline`; apply `lower_error_guard` to `LOWER_GUARDS`; apply `candidate >= baseline-max(.05*abs(baseline),1e-6)` to `HIGHER_GUARDS`; require norm `[.5,1.5]` and Pearson `.80`. No gate accepts runtime threshold overrides.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_screen.py -q
git add src/fno_acoustic/ais_screen.py tests/test_ais_screen.py
git commit -m "feat: add deterministic AIS screen gates"
```

## Task 9: Serial screen orchestrator and manifest

**Files:**
- Create: `scripts/run_ais_v2_screen.py`
- Create: `tests/test_ais_v2_cli.py`

- [ ] **Step 1: Write RED dry-run test**

```python
def test_dry_run_orders_candidates_and_never_mentions_test_split(tmp_path):
    plan = build_screen_plan(CONFIG_DIR, tmp_path)
    assert [x.candidate_id for x in plan.gate_o] == [f'N{i}' for i in range(8)]
    assert all('test' not in ' '.join(x.command).lower() for x in plan.all_commands)
    assert all('--device' in x.command and 'cuda' in x.command for x in plan.all_commands)
```

- [ ] **Step 2: Run RED**

Expected: import failure for `scripts.run_ais_v2_screen`.

- [ ] **Step 3: Implement the state machine**

The orchestrator options are `--config-dir`, `--output-dir`, `--device cuda`, `--resume`, and `--dry-run`. A fresh invocation executes smokes/Gate O and stops after publishing its decision; `--resume` executes the next approved H1/H2/H3 states until the next review boundary. Add training-only `--overfit-sample-id` and `--overfit-site-manifest`; they are legal only with `--max-train-batches 400`, must match the config-bound train ID/site SHA, and replace validation by that one train scene. Launch one child at a time, reject a second live GPU child, validate exit/artifact hashes, call pure gates, and atomically write `screen_manifest.json`. Never construct a test command.

- [ ] **Step 4: Write RED idempotence/tamper tests**

```python
def test_resume_skips_hash_valid_complete_candidate(tmp_path):
    manifest = completed_candidate_manifest(tmp_path, 'N0', 600)
    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)
    assert all(not (x.candidate_id == 'N0' and x.stop_update == 600)
               for x in plan.all_commands)

def test_resume_rejects_changed_checkpoint_or_metrics(tmp_path):
    manifest = completed_candidate_manifest(tmp_path, 'N0', 600)
    (tmp_path/'h1/N0/checkpoints/last.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)

def test_manifest_records_source_config_stats_split_gpu_and_budget(tmp_path):
    row = publish_fake_candidate(tmp_path)
    assert {'source_commit','config_sha256','normalization_stats_sha256',
            'split_sha256','gpu_name','optimizer_updates'} <= set(row)
```

- [ ] **Step 5: Implement the artifact census**

Record command, source commit, config/split/stats/best/last/metrics hashes, seed, parent, update, parameter count, peak memory, wall seconds, GPU name, and decision. A mismatch is fatal and never overwritten.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_v2_cli.py tests/test_ais_screen.py tests/test_ais_v2_configs.py -q
git add scripts/run_ais_v2_screen.py tests/test_ais_v2_cli.py
git commit -m "feat: orchestrate serial AIS v2 screening"
```

## Task 10: Complete CPU regression and final code review

**Files:**
- Modify only files required by observed failures from Tasks 1--9.

- [ ] **Step 1: Run the complete v2 suite**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest \
 tests/test_ais_normalization.py tests/test_standardized_hh_loss.py \
 tests/test_ais_checkpoint_normalization.py tests/test_ais_model_variants.py \
 tests/test_ais_v2_configs.py tests/test_ais_category_metrics.py \
 tests/test_ais_screen.py tests/test_ais_v2_cli.py -q
```

Expected: all pass.

- [ ] **Step 2: Run existing AIS regressions**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest \
 tests/test_ais_query_data.py tests/test_nonuniform_temporal_operator.py \
 tests/test_ais_sampler.py tests/test_hansen_hurwitz_loss.py tests/test_ais_mqfno.py \
 tests/test_query_reconstruction.py tests/test_ais_checkpoint_resume.py \
 tests/test_ais_query_training.py tests/test_long_horizon_metrics.py \
 tests/test_query_census.py tests/test_native400_accuracy_gate.py \
 tests/test_ais_configs.py tests/test_ais_sealed_test.py -q
```

- [ ] **Step 3: Compile and lint**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m compileall -q src/fno_acoustic \
 scripts/train_ais_mqfno.py scripts/evaluate_ais_mqfno.py \
 scripts/generate_ais_v2_configs.py scripts/run_ais_v2_screen.py
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m ruff check \
 src/fno_acoustic/ais_normalization.py src/fno_acoustic/ais_model_components.py \
 src/fno_acoustic/ais_screen.py scripts/generate_ais_v2_configs.py scripts/run_ais_v2_screen.py
```

- [ ] **Step 4: Commit integration corrections only when needed**

```bash
git add src/fno_acoustic/ais_normalization.py src/fno_acoustic/query_losses.py \
 src/fno_acoustic/query_training.py src/fno_acoustic/ais_model_components.py \
 src/fno_acoustic/model_ais_mqfno.py src/fno_acoustic/query_census.py \
 src/fno_acoustic/long_horizon_metrics.py src/fno_acoustic/ais_screen.py \
 scripts/train_ais_mqfno.py scripts/evaluate_ais_mqfno.py \
 scripts/generate_ais_v2_configs.py scripts/run_ais_v2_screen.py
git commit -m "fix: integrate AIS v2 screening contracts"
```

Skip Step 4 when no correction was required. Dispatch one final reviewer across all Task 1--10 commits; resolve every issue before CUDA execution.

## Task 11: Eight serial CUDA smokes and Gate O

**Files:**
- Create runtime artifacts only under `artifacts/ais_mqfno_zero_collapse_v2_20260714/`.

- [ ] **Step 1: Verify exclusive GPU capacity**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Expected: no compute process.

- [ ] **Step 2: Run all one-update smokes serially**

```bash
for id in n0_norm n1_wide n2_spatial n3_temporal n4_large_local n5_multi_local n6_dispersion n7_multi_dispersion; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python \
    scripts/train_ais_mqfno.py --config "configs/ais_zero_collapse_v2/${id}.yaml" \
    --device cuda --max-train-batches 1 --max-val-batches 1 \
    --output-dir "artifacts/ais_mqfno_zero_collapse_v2_20260714/smoke/${id}" || exit 1
done
```

Expected: schema-v4, full160, finite loss/gradient/prediction, and peak reserved memory below 24 GB for every candidate.

- [ ] **Step 3: Run Gate O**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python \
 scripts/run_ais_v2_screen.py --config-dir configs/ais_zero_collapse_v2 \
 --output-dir artifacts/ais_mqfno_zero_collapse_v2_20260714 --device cuda
```

The runner stops when N0 fails or fewer than four pass. Do not modify thresholds after results.

- [ ] **Step 4: Independently review Gate O artifacts**

Check fixed site IDs, exact update-400 last hashes, physical metrics, norm ratios, correlations, finite flags, and rejection reasons.

## Task 12: Execute H1/H2/H3 and report

**Files:**
- Create: `reports/ais_mqfno_zero_collapse_v2_screen.md`
- Update runtime manifest under `artifacts/ais_mqfno_zero_collapse_v2_20260714/`.

- [ ] **Step 1: Resume after approved Gate O**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python \
 scripts/run_ais_v2_screen.py --config-dir configs/ais_zero_collapse_v2 \
 --output-dir artifacts/ais_mqfno_zero_collapse_v2_20260714 --device cuda --resume
```

Expected: H1 restarts survivors; H2 resumes exactly four update-600 last checkpoints; H3 resumes exactly two update-1500 last checkpoints.

- [ ] **Step 2: Evaluate the pinned raw B1 validation baseline**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python \
 scripts/evaluate_ais_mqfno.py --config configs/ais_mqfno_64x160_b1_uniform.yaml \
 --checkpoint artifacts/ais_mqfno_full160_native400_20260714/runs/64_b1_seed20260714/checkpoints/best.pt \
 --split val --legacy-raw-b1-baseline --device cuda \
 --output-dir artifacts/ais_mqfno_zero_collapse_v2_20260714/baseline_b1
```

Expected: pinned SHA confirmed; category metrics written; no reusable training checkpoint produced.

- [ ] **Step 3: Apply final promotion gate**

Use exact update-3000 last metrics against the pinned B1. Produce one promoted candidate or registered `no_promotion`; never select a failing least-bad model.

- [ ] **Step 4: Write the evidence report**

Include old zero-collapse diagnosis; eight parameter/memory rows; every gate/rejection; global/category field, receiver, quarter, phase, high-k and k-omega metrics; 30%/5% arithmetic; all hashes; and explicit confirmation that test IDs remain unopened.

- [ ] **Step 5: Verify and commit only the report**

```bash
PYTHONPATH=src /home/jiayh/miniforge3/envs/qwen/bin/python -m pytest tests/test_ais_screen.py tests/test_ais_v2_cli.py -q
git add reports/ais_mqfno_zero_collapse_v2_screen.md
git commit -m "docs: report AIS v2 network screen"
```

Do not add checkpoints or mutable runtime artifacts to git.

## Required per-task review protocol

For Tasks 1--9: dispatch one fresh implementer with complete task text and TDD; require commit and self-review; dispatch a spec reviewer; resolve all issues; only then dispatch a code-quality reviewer; resolve all issues and rerun focused tests before marking the task complete. After Task 10, perform final cross-implementation review. Tasks 11--12 remain serial and receive artifact review after each gate.
