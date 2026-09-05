# V61 Multiscale Band Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate a sparse family-routed, multiscale spectral low/mid-band adapter that exactly preserves the V49 high-band anchor and exceeds the registered 20% three-record all-time reduction gate.

**Architecture:** Extend the backwards-compatible V4 probe schema with explicit adapter architecture fields.  V61 reuses the frozen decoder shared tensor, dispatches only the selected family expert, fuses full-resolution and pooled spectral/local stacks with exact-time FiLM, then applies the existing free-surface-aware low/mid projection before adding the frozen V49 anchor.

**Tech Stack:** Python 3.11, PyTorch, pytest, YAML/JSON evidence, four-GPU `torchrun`/NCCL, SSH/rsync.

---

## File map

- `saved_time_phase_operator_v4/probe.py`: immutable architecture schema and legacy-safe defaults.
- `saved_time_phase_operator_v4/band_adapter.py`: V60 legacy expert factory, V61 multiscale expert, sparse routing, and hard frequency guard.
- `saved_time_phase_operator_v4/decoder.py`: construct the selected adapter from explicit fields.
- `saved_time_phase_operator_v4/operator.py`: thread adapter fields into the dense decoder; output safety path remains unchanged.
- `scripts/train_saved_time_v4_probe.py`: map `ProbeVariant` fields into the model constructor.
- `scripts/train_saved_time_v4_full_support.py`: validate overrides and audit strict checkpoint expansion/zero identity.
- `scripts/prepare_saved_time_band_adapter_candidate.py`: generate identity-bound V61 diagnostic/pilot configurations.
- `tests/saved_time_phase_operator_v4/test_band_adapter.py`: architecture, gradients, routing, projection, and transfer tests.
- `tests/saved_time_phase_operator_v4/test_operator.py`: end-to-end parent identity and full-field contract tests.
- `tests/saved_time_phase_operator_v4/test_full_support_runner.py`: configuration override validation tests.
- `tests/saved_time_phase_operator_v4/test_band_adapter_cli.py`: generated V61 configuration/provenance tests.
- `configs/saved_time_v4/generated/saved_time_v61_multiscale_band_adapter_overfit_r1.yaml`: generated three-record diagnostic configuration.
- `docs/superpowers/reports/2026-07-20-v61-multiscale-band-adapter-progress.md`: measured gate and resource evidence.

### Task 1: Register a legacy-safe adapter architecture schema

**Files:**
- Modify: `saved_time_phase_operator_v4/probe.py`
- Modify: `scripts/train_saved_time_v4_full_support.py`
- Test: `tests/saved_time_phase_operator_v4/test_full_support_runner.py`

- [ ] **Step 1: Write failing schema and override-validation tests**

```python
def test_multiscale_band_adapter_overrides_are_registered():
    identity = _parent_identity()
    config = {
        "variant_overrides": {
            "band_adapter_rank": 32,
            "band_adapter_architecture": "multiscale_spectral",
            "band_adapter_spectral_rank": 32,
            "band_adapter_modes": 32,
            "band_adapter_full_depth": 4,
            "band_adapter_coarse_depth": 2,
            "band_adapter_activation_checkpointing": True,
        }
    }
    variant = probe_variant_for_config(config, identity)
    assert variant.band_adapter_architecture == "multiscale_spectral"
    assert variant.band_adapter_rank == 32
    assert variant.band_adapter_spectral_rank == 32
    assert variant.band_adapter_modes == 32
    assert variant.band_adapter_full_depth == 4
    assert variant.band_adapter_coarse_depth == 2
    assert variant.band_adapter_activation_checkpointing is True


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("band_adapter_architecture", "unknown"),
        ("band_adapter_spectral_rank", 0),
        ("band_adapter_modes", 102),
        ("band_adapter_full_depth", 0),
        ("band_adapter_coarse_depth", 0),
    ),
)
def test_multiscale_band_adapter_rejects_invalid_overrides(name, value):
    config = {"variant_overrides": {
        "band_adapter_rank": 32,
        "band_adapter_architecture": "multiscale_spectral",
        "band_adapter_spectral_rank": 32,
        "band_adapter_modes": 32,
        "band_adapter_full_depth": 4,
        "band_adapter_coarse_depth": 2,
        name: value,
    }}
    with pytest.raises(ValueError, match="band adapter"):
        probe_variant_for_config(config, _parent_identity())
```

- [ ] **Step 2: Run the focused tests and observe the missing-field failure**

Run: `pytest -q tests/saved_time_phase_operator_v4/test_full_support_runner.py -k multiscale_band_adapter`

Expected: FAIL because `ProbeVariant` and the override allowlist do not contain the V61 fields.

- [ ] **Step 3: Add schema defaults and complete validation**

```python
@dataclass(frozen=True)
class ProbeVariant:
    depth: int
    use_local_phase: bool
    spectral_rank: int = 112
    modes: int = 32
    coupled_axes: bool = False
    local_differential_residual: bool = False
    coupled_2d_rank: int = 0
    temporal_basis_rank: int = 0
    family_expert_rank: int = 0
    band_adapter_rank: int = 0
    band_adapter_architecture: str = "low_rank"
    band_adapter_spectral_rank: int = 32
    band_adapter_modes: int = 32
    band_adapter_full_depth: int = 4
    band_adapter_coarse_depth: int = 2
    band_adapter_activation_checkpointing: bool = True
```

In `probe_variant_for_config`, admit the six new names and enforce architecture
membership `{"low_rank", "multiscale_spectral"}`, ranks/depths in `[1, 128]`,
and modes in `[1, 101]`; require `band_adapter_activation_checkpointing` to be a
boolean.  Also reject a non-default multiscale configuration when
`band_adapter_rank == 0`.

- [ ] **Step 4: Run schema tests**

Run: `pytest -q tests/saved_time_phase_operator_v4/test_full_support_runner.py -k 'variant or multiscale_band_adapter'`

Expected: PASS.

- [ ] **Step 5: Commit schema support**

```bash
git add saved_time_phase_operator_v4/probe.py scripts/train_saved_time_v4_full_support.py tests/saved_time_phase_operator_v4/test_full_support_runner.py
git commit -m "feat: register V61 band adapter schema"
```

### Task 2: Implement the multiscale spectral residual expert

**Files:**
- Modify: `saved_time_phase_operator_v4/band_adapter.py`
- Test: `tests/saved_time_phase_operator_v4/test_band_adapter.py`

- [ ] **Step 1: Write failing expert behavior tests**

```python
def _multiscale_adapter():
    return BandLimitedFamilyAdapter(
        width=8,
        rank=6,
        architecture="multiscale_spectral",
        spectral_rank=4,
        modes=4,
        full_depth=2,
        coarse_depth=1,
        activation_checkpointing=False,
    )


def test_multiscale_adapter_is_exact_zero_then_wakes_internal_gradients():
    adapter = _multiscale_adapter()
    shared = torch.randn(6, 8, 17, 19)
    time_features = torch.randn(2, 3, 5)
    mapping = torch.tensor([0, 1])
    routes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    initial = adapter(shared, time_features, mapping, routes)
    assert initial.shape == (2, 3, 17, 19)
    assert torch.count_nonzero(initial) == 0
    initial.sum().backward()
    assert adapter.experts[0].output.weight.grad.norm() > 0
    with torch.no_grad():
        adapter.experts[0].output.weight.normal_(std=1.0e-3)
    adapter.zero_grad(set_to_none=True)
    adapter(shared, time_features, mapping, routes).square().mean().backward()
    assert adapter.experts[0].input_projection.weight.grad.norm() > 0
    assert adapter.experts[0].full_stack.blocks[0].residual_scale.grad.norm() > 0


def test_multiscale_adapter_sparse_one_hot_dispatch_skips_unused_family():
    adapter = _multiscale_adapter()
    calls = [0, 0, 0]
    handles = [expert.register_forward_hook(
        lambda _m, _i, _o, index=index: calls.__setitem__(index, calls[index] + 1)
    ) for index, expert in enumerate(adapter.experts)]
    try:
        adapter(
            torch.randn(4, 8, 15, 17),
            torch.randn(2, 2, 5),
            torch.tensor([0, 1]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [1, 0, 1]


def test_multiscale_adapter_soft_route_matches_explicit_weighted_sum():
    adapter = _multiscale_adapter()
    with torch.no_grad():
        for expert in adapter.experts:
            expert.output.weight.normal_(std=1.0e-3)
    shared = torch.randn(2, 8, 15, 17)
    features = torch.randn(1, 2, 5)
    probabilities = torch.tensor([[0.2, 0.3, 0.5]])
    observed = adapter(shared, features, torch.tensor([0]), probabilities)
    expected = sum(
        probabilities[0, index] * expert(shared, features)
        for index, expert in enumerate(adapter.experts)
    )
    torch.testing.assert_close(observed, expected)
```

- [ ] **Step 2: Run tests and observe constructor failures**

Run: `pytest -q tests/saved_time_phase_operator_v4/test_band_adapter.py -k multiscale`

Expected: FAIL because the architecture keyword and expert do not exist.

- [ ] **Step 3: Implement the expert and backwards-compatible factory**

```python
class MultiscaleSpectralResidualExpert(nn.Module):
    def __init__(self, *, width, latent_width, spectral_rank, modes,
                 full_depth, coarse_depth, activation_checkpointing):
        super().__init__()
        self.latent_width = int(latent_width)
        self.input_projection = nn.Conv2d(width, latent_width, 1)
        self.full_stack = FactorizedComplexResidualStack(
            latent_width, spectral_rank, modes, full_depth,
            activation_checkpointing=activation_checkpointing,
        )
        self.coarse_stack = FactorizedComplexResidualStack(
            latent_width, spectral_rank, modes, coarse_depth,
            activation_checkpointing=activation_checkpointing,
        )
        self.fuse = nn.Conv2d(2 * latent_width, latent_width, 1)
        self.time = nn.Sequential(
            nn.Linear(5, latent_width), nn.GELU(),
            nn.Linear(latent_width, 2 * latent_width),
        )
        self.output = nn.Conv2d(latent_width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, shared, time_features):
        records, times = time_features.shape[:2]
        lifted = self.input_projection(shared)
        full = self.full_stack(lifted)
        coarse = F.avg_pool2d(lifted, kernel_size=2, stride=2, ceil_mode=True)
        coarse = self.coarse_stack(coarse)
        coarse = F.interpolate(coarse, size=full.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.fuse(torch.cat((full, coarse), dim=1))
        scale, shift = self.time(time_features).reshape(
            records * times, 2 * self.latent_width, 1, 1
        ).chunk(2, dim=1)
        hidden = F.gelu(hidden * (1.0 + scale) + shift)
        return self.output(hidden).reshape(records, times, *shared.shape[-2:])
```

`BandLimitedFamilyAdapter` selects `LowRankResidualExpert` only for
`architecture="low_rank"`; otherwise it creates three instances of the class
above.  Keep `_routed_expert_mix` unchanged so exact one-hot routes remain sparse
and soft routes remain a differentiable mixture.

- [ ] **Step 4: Run adapter tests**

Run: `pytest -q tests/saved_time_phase_operator_v4/test_band_adapter.py`

Expected: PASS, including all legacy V60 tests.

- [ ] **Step 5: Commit the expert**

```bash
git add saved_time_phase_operator_v4/band_adapter.py tests/saved_time_phase_operator_v4/test_band_adapter.py
git commit -m "feat: add multiscale spectral band experts"
```

### Task 3: Thread V61 fields through the full operator

**Files:**
- Modify: `saved_time_phase_operator_v4/decoder.py`
- Modify: `saved_time_phase_operator_v4/operator.py`
- Modify: `scripts/train_saved_time_v4_probe.py`
- Test: `tests/saved_time_phase_operator_v4/test_operator.py`

- [ ] **Step 1: Write a failing full-operator identity/safety test**

```python
def test_multiscale_band_adapter_preserves_parent_full_field_and_high_band():
    torch.manual_seed(20260720)
    parent = _model(family_expert_rank=4, band_adapter_rank=0).eval()
    candidate = _model(
        family_expert_rank=4,
        band_adapter_rank=6,
        band_adapter_architecture="multiscale_spectral",
        band_adapter_spectral_rank=4,
        band_adapter_modes=4,
        band_adapter_full_depth=2,
        band_adapter_coarse_depth=1,
    ).eval()
    missing = candidate.load_state_dict(parent.state_dict(), strict=False).missing_keys
    assert missing and all(key.startswith("dense_decoder.band_limited_adapter.") for key in missing)
    parent_wavefield = _forward(parent)
    candidate_wavefield = _forward(candidate)
    torch.testing.assert_close(candidate_wavefield, parent_wavefield, rtol=0.0, atol=0.0)
    assert candidate_wavefield.shape[-2:] == (201, 201)
    with torch.no_grad():
        for expert in candidate.dense_decoder.band_limited_adapter.experts:
            expert.output.weight.normal_(std=1.0e-3)
    adapted = _forward(candidate)
    parent_fft = torch.fft.rfft2(parent_wavefield.float(), norm="ortho")
    adapted_fft = torch.fft.rfft2(adapted.float(), norm="ortho")
    high = registered_high_band_mask(201, 201, adapted.device)
    assert (adapted_fft[..., high] - parent_fft[..., high]).abs().max() < 2.0e-5
    assert adapted[..., 0, :].abs().max() < 2.0e-5
```

- [ ] **Step 2: Run the test and observe constructor failure**

Run: `pytest -q tests/saved_time_phase_operator_v4/test_operator.py -k multiscale_band_adapter`

Expected: FAIL because operator/decoder constructors do not accept V61 fields.

- [ ] **Step 3: Add constructor plumbing without changing the output path**

Add the six adapter arguments to `SavedTimePhaseOperatorV4` and
`PropagationConditionedDenseDecoder`, then construct:

```python
self.band_limited_adapter = (
    None if adapter_rank == 0 else BandLimitedFamilyAdapter(
        width=width,
        rank=adapter_rank,
        architecture=band_adapter_architecture,
        spectral_rank=band_adapter_spectral_rank,
        modes=band_adapter_modes,
        full_depth=band_adapter_full_depth,
        coarse_depth=band_adapter_coarse_depth,
        activation_checkpointing=band_adapter_activation_checkpointing,
    )
)
```

Map every field from `ProbeVariant` in `scripts/train_saved_time_v4_probe.py::_model`.
Do not edit `SavedTimePhaseOperatorV4._dense_block_with_anchor_increment_and_routing`:
it must continue applying `free_surface_factor` before
`project_low_mid_increment` and adding the frozen anchor.

- [ ] **Step 4: Run operator, adapter, and probe tests**

Run: `pytest -q tests/saved_time_phase_operator_v4/test_operator.py tests/saved_time_phase_operator_v4/test_band_adapter.py tests/saved_time_phase_operator_v4/test_probe.py`

Expected: PASS.

- [ ] **Step 5: Commit operator plumbing**

```bash
git add saved_time_phase_operator_v4/decoder.py saved_time_phase_operator_v4/operator.py scripts/train_saved_time_v4_probe.py tests/saved_time_phase_operator_v4/test_operator.py
git commit -m "feat: wire V61 adapter into saved-time operator"
```

### Task 4: Make strict transfer and generated configuration architecture-aware

**Files:**
- Modify: `scripts/train_saved_time_v4_full_support.py`
- Modify: `scripts/prepare_saved_time_band_adapter_candidate.py`
- Test: `tests/saved_time_phase_operator_v4/test_band_adapter.py`
- Test: `tests/saved_time_phase_operator_v4/test_band_adapter_cli.py`

- [ ] **Step 1: Write failing identity and generation tests**

```python
def test_multiscale_adapter_identity_report_records_architecture_and_parameters():
    model = nn.Module()
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = _multiscale_adapter()
    report = band_adapter_identity_report(
        model, expected_rank=6, expected_architecture="multiscale_spectral"
    )
    assert report["band_adapter_architecture"] == "multiscale_spectral"
    assert report["band_adapter_rank"] == 6
    assert report["trainable_parameters"] == sum(
        parameter.numel() for parameter in model.dense_decoder.band_limited_adapter.parameters()
    )
    assert report["exact_parent_identity"] is True


def test_build_v61_candidate_registers_multiscale_fields(tmp_path):
    config = build_band_adapter_pilot_config(
        _v49_identity(),
        parent_checkpoint=tmp_path / "epoch_0001.pt",
        parent_checkpoint_identity=tmp_path / "run_identity.json",
        artifact_dir=tmp_path / "v61",
        band_adapter_rank=32,
        band_adapter_architecture="multiscale_spectral",
        band_adapter_spectral_rank=32,
        band_adapter_modes=32,
        band_adapter_full_depth=4,
        band_adapter_coarse_depth=2,
        band_adapter_activation_checkpointing=True,
    )
    assert config["variant_overrides"]["band_adapter_architecture"] == "multiscale_spectral"
    assert config["variant_overrides"]["band_adapter_full_depth"] == 4
    assert config["variant_overrides"]["band_adapter_activation_checkpointing"] is True
    assert config["band_limited_adapter"]["cutoff_normalized_radius"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run focused tests and observe signature failures**

Run: `pytest -q tests/saved_time_phase_operator_v4/test_band_adapter.py tests/saved_time_phase_operator_v4/test_band_adapter_cli.py -k multiscale`

Expected: FAIL because the transfer audit and candidate builder do not accept architecture fields.

- [ ] **Step 3: Generalize the identity audit and candidate builder**

Add a small helper that reports either `expert.down.out_channels` for legacy
experts or `expert.latent_width` for V61.  Require all three experts to match the
registered architecture/rank, require every `expert.output` tensor to be zero,
and record the exact trainable parameter count.  Pass
`expected_architecture=variant.band_adapter_architecture` from `_load_parent_model`.

Extend the candidate builder and CLI with exact defaults:

```python
parser.add_argument("--band-adapter-architecture", choices=("low_rank", "multiscale_spectral"), default="low_rank")
parser.add_argument("--band-adapter-spectral-rank", type=int, default=32)
parser.add_argument("--band-adapter-modes", type=int, default=32)
parser.add_argument("--band-adapter-full-depth", type=int, default=4)
parser.add_argument("--band-adapter-coarse-depth", type=int, default=2)
```

Legacy V60 generation must produce its previous dictionary byte-for-byte except
for YAML serialization order.  V61 writes the new fields only when
`band_adapter_architecture == "multiscale_spectral"`, including
`band_adapter_activation_checkpointing: true`.

- [ ] **Step 4: Run transfer and CLI tests**

Run: `pytest -q tests/saved_time_phase_operator_v4/test_band_adapter.py tests/saved_time_phase_operator_v4/test_band_adapter_cli.py tests/saved_time_phase_operator_v4/test_full_support_runner.py`

Expected: PASS.

- [ ] **Step 5: Commit transfer/config support**

```bash
git add scripts/train_saved_time_v4_full_support.py scripts/prepare_saved_time_band_adapter_candidate.py tests/saved_time_phase_operator_v4/test_band_adapter.py tests/saved_time_phase_operator_v4/test_band_adapter_cli.py
git commit -m "feat: register V61 transfer and candidate config"
```

### Task 4A: Separate zero-head and spectral-feature learning rates

**Files:**
- Modify: `saved_time_phase_operator_v4/band_adapter.py`
- Modify: `scripts/diagnose_saved_time_temporal_three_record_overfit.py`
- Modify: `scripts/train_saved_time_v4_full_support.py`
- Modify: `scripts/prepare_saved_time_band_adapter_candidate.py`
- Test: `tests/saved_time_phase_operator_v4/test_band_adapter.py`
- Test: `tests/saved_time_phase_operator_v4/test_temporal_three_record_overfit.py`

- [x] **Step 1: Write failing optimizer partition tests**

```python
def test_band_adapter_optimizer_separates_output_head_and_features():
    model = nn.Module()
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = _multiscale_adapter()
    optimizer = build_band_adapter_adamw(
        model,
        feature_lr=3.0e-4,
        output_lr=1.0e-5,
        weight_decay=1.0e-6,
    )
    observed = {
        id(parameter): (group["group_name"], group["lr"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    named = dict(model.named_parameters())
    assert set(observed) == {id(parameter) for parameter in named.values()}
    for name, parameter in named.items():
        group, learning_rate = observed[id(parameter)]
        if ".output." in name:
            assert group.startswith("adapter_output_")
            assert learning_rate == pytest.approx(1.0e-5)
        else:
            assert group.startswith("adapter_feature_")
            assert learning_rate == pytest.approx(3.0e-4)
```

- [x] **Step 2: Run the focused test and observe the missing builder failure**

Run: `CUDA_VISIBLE_DEVICES='' /home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_band_adapter.py -k optimizer_separates`

Expected: FAIL because `build_band_adapter_adamw` is not defined.

- [x] **Step 3: Implement a disjoint adapter-only AdamW builder**

```python
def build_band_adapter_adamw(model, *, feature_lr, output_lr, weight_decay,
                              implementation="single_tensor"):
    from .full_support import adamw_backend_options

    rates = {
        "adapter_feature": float(feature_lr),
        "adapter_output": float(output_lr),
    }
    if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
        raise ValueError("band adapter learning rates must be positive and finite")
    if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0:
        raise ValueError("band adapter weight decay must be finite and nonnegative")
    path = model.dense_decoder.band_limited_adapter
    groups = {
        name: {"decay": [], "no_decay": []}
        for name in ("adapter_feature", "adapter_output")
    }
    for name, parameter in model.named_parameters():
        if not name.startswith("dense_decoder.band_limited_adapter."):
            continue
        role = "adapter_output" if ".output." in name else "adapter_feature"
        decay = "no_decay" if parameter.ndim <= 1 or name.endswith("bias") else "decay"
        groups[role][decay].append(parameter)
    if any(not any(parts.values()) for parts in groups.values()):
        raise ValueError("band adapter optimizer roles must both be nonempty")
    parameter_groups = []
    for role in ("adapter_feature", "adapter_output"):
        for decay_class in ("decay", "no_decay"):
            parameters = groups[role][decay_class]
            if parameters:
                parameter_groups.append({
                    "params": parameters,
                    "lr": rates[role],
                    "initial_lr": rates[role],
                    "group_name": f"{role}_{decay_class}",
                    "weight_decay": float(weight_decay) if decay_class == "decay" else 0.0,
                })
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=0.0,
        **adamw_backend_options(implementation),
    )
```

The implementation rejects nonpositive/nonfinite rates, a missing adapter,
empty roles, duplicate parameters, or parameters outside the adapter prefix.

- [x] **Step 4: Select the builder only for an explicitly registered split**

When both optimizer keys below exist, the diagnostic and full-support runner use
the adapter-only builder; otherwise they retain `build_staged_adamw` unchanged:

```yaml
optimizer:
  band_adapter_feature_learning_rate: 0.0003
  band_adapter_output_learning_rate: 0.00001
```

The candidate generator writes both keys only for `multiscale_spectral`.
Diagnostic CLI overrides are
`--band-adapter-feature-learning-rate` and
`--band-adapter-output-learning-rate`; run identity records both exact values.

- [x] **Step 5: Run focused and full V4 tests**

Run: `CUDA_VISIBLE_DEVICES='' /home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_band_adapter.py tests/saved_time_phase_operator_v4/test_temporal_three_record_overfit.py tests/saved_time_phase_operator_v4/test_full_support_runner.py`

Expected: PASS, followed by all 567-or-more V4 tests passing.

- [x] **Step 6: Commit the optimizer split**

```bash
git add saved_time_phase_operator_v4/band_adapter.py scripts/diagnose_saved_time_temporal_three_record_overfit.py scripts/train_saved_time_v4_full_support.py scripts/prepare_saved_time_band_adapter_candidate.py tests/saved_time_phase_operator_v4/test_band_adapter.py tests/saved_time_phase_operator_v4/test_temporal_three_record_overfit.py docs/superpowers/specs/2026-07-20-v61-multiscale-band-adapter-design.md docs/superpowers/plans/2026-07-20-v61-multiscale-band-adapter.md
git commit -m "feat: split V61 head and feature learning rates"
```

### Task 5: Verify the complete local regression suite and generate V61 evidence

**Files:**
- Create: `configs/saved_time_v4/generated/saved_time_v61_multiscale_band_adapter_overfit_r1.yaml`
- Create: `docs/superpowers/reports/2026-07-20-v61-multiscale-band-adapter-progress.md`

- [x] **Step 1: Run all V4 tests**

Run: `pytest -q tests/saved_time_phase_operator_v4`

Expected: all tests PASS with no legacy V60 regression.

- [x] **Step 2: Generate the identity-bound V61 configuration**

Run the existing preparation CLI with the exact registered V49 checkpoint and
identity paths from the V60 selection report, plus:

```bash
python scripts/prepare_saved_time_band_adapter_candidate.py \
  --parent-checkpoint /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v49_family_experts_structural_prior_pilot_r1/pilot/checkpoints/epoch_0001.pt \
  --parent-checkpoint-identity /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v49_family_experts_structural_prior_pilot_r1/pilot/run_identity.json \
  --dataset-h5 /home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5 \
  --output-config configs/saved_time_v4/generated/saved_time_v61_multiscale_band_adapter_overfit_r1.yaml \
  --artifact-dir /root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v61_multiscale_band_adapter_r1 \
  --report docs/superpowers/reports/2026-07-20-v61-parent-selection.json \
  --band-adapter-rank 32 \
  --band-adapter-architecture multiscale_spectral \
  --band-adapter-spectral-rank 32 \
  --band-adapter-modes 32 \
  --band-adapter-full-depth 4 \
  --band-adapter-coarse-depth 2 \
  --physical-microbatch-records 24 \
  --macro-records 24 \
  --effective-batch-records 96 \
  --training-frames-per-record 24 \
  --dense-learning-rate 3e-4 \
  --band-adapter-feature-learning-rate 3e-4 \
  --band-adapter-output-learning-rate 1e-5
```

Verify config SHA256, checkpoint SHA256, run digest, manifest digest,
time-axis SHA256, one-source contract, 401 stored times, and 201 output points
per spatial axis against existing provenance files.

- [x] **Step 3: Record the V60 rejection and V61 registration**

Write the report with exact measured V60 values, V61 parameter count, all test
counts, generated config hash, and the unchanged data/boundary contract.

- [x] **Step 4: Commit generated configuration and report**

```bash
git add configs/saved_time_v4/generated/saved_time_v61_multiscale_band_adapter_overfit_r1.yaml docs/superpowers/reports/2026-07-20-v61-multiscale-band-adapter-progress.md docs/superpowers/reports/2026-07-20-v61-parent-selection.json
git commit -m "experiment: register V61 multiscale adapter gate"
```

### Task 6: Run the remote overfit, resource, and promotion gates

**Files:**
- Remote artifacts: `/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v61_multiscale_band_adapter_r1`
- Local artifacts: `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v61_multiscale_band_adapter_r1`
- Modify: `docs/superpowers/reports/2026-07-20-v61-multiscale-band-adapter-progress.md`

- [ ] **Step 1: Sync only committed V61 source/config files and verify hashes**

Use `rsync -aR` without `--delete`, then compare SHA256 for every transferred
file.  Run the focused remote tests before allocating GPUs.

- [ ] **Step 2: Launch the three-record staggered-time diagnostic detached**

Run with `nohup setsid`, one uniform/corrected-layered/Marmousi training record,
64 exact-time appearances per update, 200 updates, evaluation every 20 updates,
and all-401-time terminal evaluation.  The command uses
`scripts/diagnose_saved_time_temporal_three_record_overfit.py`, the V61 config,
V49 parent checkpoint/identity, equal family weights, and physical diagnostic
microbatch 3.  Before the 200-update gate, sweep feature rates `1e-4`, `3e-4`,
`1e-3`, and `3e-3` for 20 updates from the same V49 anchor while holding the
output-head rate at `1e-5`; promote only the best finite fixed-panel result.

Expected terminal evidence:

```json
{
  "status": "complete",
  "updates_completed": 200,
  "anchor_relative_reduction": 0.20,
  "all_saved_metrics": {
    "high_band_anchor_delta": 0.00002
  }
}
```

The actual reduction must be at least `0.20`; the actual high-band delta must be
at most `2e-5`; every family must improve over its V49 anchor.

- [ ] **Step 3: Reject or promote from terminal evidence**

If any registered threshold fails, retain/sync the best checkpoint and record
V61 as rejected without starting a long job.  If all thresholds pass, run the
same-panel 48-record sampled-time and all-401-time evaluations from the selected
checkpoint.

- [ ] **Step 4: Measure the four-GPU batch-24 resource smoke**

Launch one four-GPU update with macro/microbatch 24 per rank, 24 frames, four
macros per update, and effective global batch 96.  Require all ranks active,
finite loss/gradients, peak allocated memory below 23 GiB per rank, no OOM, and
an identity-bound checkpoint/log.

- [ ] **Step 5: Launch only an admitted short curriculum pilot**

Use uniform, corrected layered, then Marmousi stages with balanced replay.  Save
every epoch checkpoint and intermediate-step metrics.  The external gate must
select a complete epoch that improves aggregate and every-family metrics while
preserving the high band and resource constraints.

- [ ] **Step 6: Synchronize all evidence locally and update the report**

Use resumable `rsync -a --partial --append-verify` for checkpoint trees and logs.
Verify SHA256 for selected checkpoint, `best.pt`, `latest.pt`, run identity,
terminal report, metrics JSONL, and launch log.  Record remote PID lifecycle,
GPU peak memory/utilization/power, elapsed time, exact selected checkpoint, and
gate decision in the progress report.

- [ ] **Step 7: Commit the measured gate report**

```bash
git add docs/superpowers/reports/2026-07-20-v61-multiscale-band-adapter-progress.md
git commit -m "docs: record V61 multiscale adapter gate"
```

### Task 7: Admit long curriculum training only after V61 passes

**Files:**
- Create: `configs/saved_time_v4/generated/saved_time_v61_multiscale_band_adapter_curriculum_r1.yaml`
- Create: `docs/superpowers/reports/2026-07-20-v61-curriculum-launch.md`

- [ ] **Step 1: Bind the long configuration to the admitted checkpoint**

Set the parent checkpoint to the selected, hash-verified V61 pilot checkpoint;
retain one source per record, no receiver input, the corrected vertical layered
family, exact stored-time sampling, and the 201x201 output contract.  Register
physical batch 24/rank and global effective batch 96 only if the resource smoke
passed unchanged.

- [ ] **Step 2: Audit schedule and checkpoint cadence locally and remotely**

Run the existing dry-run/schedule audit.  Require equal intended family replay,
all ranks assigned nonempty work, one checkpoint per epoch, intermediate-step
metrics, and an independent held-out evaluation panel excluded from training.

- [ ] **Step 3: Launch detached and verify survival**

Launch with `nohup setsid torchrun --standalone --nproc_per_node=4`, record PID,
then verify after the shell exits that the process group, all four CUDA ranks,
log growth, and first finite metric remain live.

- [ ] **Step 4: Monitor once per epoch and preserve the best verified state**

At each epoch, analyze aggregate/family validation trend, training/validation
gap, high-band delta, memory, throughput, and ETA.  Continue only finite,
identity-consistent epochs; keep the best complete checkpoint rather than a
worse terminal state.

- [ ] **Step 5: Apply the final independent acceptance gate**

Require neural-operator aggregate relative L2 `<0.10` and each of uniform,
layered, and Marmousi `<0.12` on the independent held-out set.  Generate and
sync velocity/source plots, true/predicted/residual full-wavefield snapshots,
and receiver traces as diagnostics only.  Instance adaptation follows only
after pretraining and may use only the first two post-onset snapshots plus
velocity/source inputs.
