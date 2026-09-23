#!/usr/bin/env python3
"""R14: forward ORACLE CEILING for variant T (time-conditioned, radially binned
spectral gain on the dense correction path).

DIAGNOSTIC, FORWARD ONLY.  No optimizer step, no training arm, no write outside
this directory.  The frozen design (selection, bin schemes, self-check gates and
the decision rule) is CEILING_DESIGN.md, written before any number here existed.

What it computes
----------------
The decoder returns  y = coarse + s*r  and the deployed field is  y*surf.
R6 measured the rank-1 optimum  min_s ||e + s r||/||e|| = sqrt(1-cos^2(r,-e))
(median 0.9984 over 144 frames).  This is its exact B-dimensional generalisation:
split r by radial wavenumber into r_1..r_B and solve, independently per frame,

    min_{g_1..g_B} || e + sum_b g_b r_b ||,    best_ratio = that norm / ||e||

g is unconstrained and free per frame, therefore free in t with no
parameterisation, which is strictly looser than variant T's g = 1 + A phi(t).
g_b also absorbs s, so a shut gate cannot depress the number.

NOT a strict upper bound on variant T: the real gate is at an INTERMEDIATE layer
with further spectral blocks, a 1x1 conv and nonlinearities after it, and is
per-channel over the retained modes only.  The two families are not nested in
either direction.  See CEILING_DESIGN.md section 6.

Calibre discipline: global calibre from run_identity.json -> effective_config
(never config.py defaults; active_horizon_s 0.60 is refused), per-record
source_t0_s from record.source_parameters[3] (never time_s[0]), windows from the
production evaluation.resolve_onset_frame / evaluation.future_window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REL = Path('/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/release_ic8_40m_20260910')
# deliberately NOT coda_round7_hinge_20260919/snapshot: another agent is editing that tree.
TREE = REL / 'research/coda_round6_density_20260919/snapshot'
ANCHOR = Path('/root/autodl-tmp/staging/l2_lwc84_ddp4_20260915_v1/formal/checkpoints/'
              'checkpoint_step_00029359.pt')
RUN_IDENTITY = Path('/root/autodl-tmp/staging/l2_lwc84_ddp4_20260915_v1/formal/run_identity.json')
PARENT_CFG = REL / 'research/l2_lwc84_ddp4_20260915_v1/train.yaml'
LISTS = REL / 'research/marmousi_longtime_levers_20260912/MEASUREMENT_LISTS_FROZEN.json'
EXCLUSIONS = REL / 'research/coda_round1_20260916/EXCLUSION_GROUPS_FROZEN.json'
R6 = Path('/root/autodl-tmp/staging/r6_alignment/ALIGNMENT_PROBE.json')
OUT = Path('/root/autodl-tmp/staging/r14_variantT_ceiling_20260921')

# eval_metrics.FREQUENCY_EDGES, re-derived from the module at runtime and asserted.
PROJECT_EDGES = (0.0, 0.005, 0.015, 0.03, float('inf'))
R6_SELFCHECK_RANGE = (0.9979, 0.9989)   # 0.9984 +/- 5e-4, frozen in CEILING_DESIGN.md
RECON_TOL = 1e-10
MONOTONE_TOL = 1e-6


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# bin schemes (pure, unit-testable)
# ---------------------------------------------------------------------------
def nested_edges(b: int, rho_max: float) -> list[float]:
    """Strict refinement hierarchy of eval_metrics.FREQUENCY_EDGES.

    B=1 (0,inf) < B=2 < B=4 (= FREQUENCY_EDGES) < B=8 < B=16.  Nesting makes the
    per-frame best_ratio monotone non-increasing in B, which is a free self-check.
    The open band [.03, inf) is subdivided on [.03, rho_max]; its last edge stays
    inf so no coefficient can ever fall outside the partition.
    """
    if b == 1:
        return [0.0, float('inf')]
    if b == 2:
        return [0.0, 0.015, float('inf')]
    if b == 4:
        return list(PROJECT_EDGES)
    if b in (8, 16):
        per = b // 4
        edges = []
        for lo, hi in zip(PROJECT_EDGES[:-1], PROJECT_EDGES[1:]):
            top = rho_max if not np.isfinite(hi) else hi
            edges.extend(list(np.linspace(lo, top, per + 1))[:-1])
        edges.append(float('inf'))
        return edges
    raise ValueError(f'no nested scheme defined for B={b}')


def equalwidth_edges(b: int, rho_max: float) -> list[float]:
    """Equal width in |k| over [0, rho_max] -- the binning variant T itself uses
    (time_spectral_gate.radial_bin_maps), as a robustness contrast."""
    edges = list(np.linspace(0.0, rho_max, b + 1))[:-1]
    edges.append(float('inf'))
    return edges


def solve_bins(columns: np.ndarray, defect: np.ndarray):
    """min_g ||defect + columns @ g|| via one thin SVD.

    Returns (g, ratio, cond, rank).  One decomposition gives the solution, the
    condition number and the numerical rank, so the conditioning is reported
    rather than assumed.  B=1 reduces exactly to s* = <-e,r>/||r||^2.
    """
    u, sv, vt = np.linalg.svd(columns, full_matrices=False)
    smax = float(sv.max()) if sv.size else 0.0
    tol = smax * max(columns.shape) * np.finfo(np.float64).eps
    keep = sv > tol
    g = np.zeros(columns.shape[1], dtype=np.float64)
    if keep.any():
        g = vt[keep].T @ ((u[:, keep].T @ (-defect)) / sv[keep])
    resid = float(np.linalg.norm(columns @ g + defect))
    den = float(np.linalg.norm(defect))
    smin = float(sv.min()) if sv.size else 0.0
    cond = smax / smin if smin > 0 else float('inf')
    return g, (resid / den if den > 0 else float('nan')), cond, int(keep.sum())


def self_test() -> int:
    fail = []
    rng = np.random.default_rng(0)
    # 1. B=1 reduces to the rank-1 alignment identity
    e = rng.normal(size=500)
    r = rng.normal(size=500)
    cos = float(-e @ r / (np.linalg.norm(e) * np.linalg.norm(r)))
    _, ratio, _, _ = solve_bins(r[:, None], e)
    if abs(ratio - np.sqrt(1 - cos ** 2)) > 1e-12:
        fail.append(f'B=1 != sqrt(1-cos^2): {ratio}')
    # 2. exact fit -> ratio 0
    cols = rng.normal(size=(400, 5))
    g_true = rng.normal(size=5)
    _, ratio, _, _ = solve_bins(cols, -(cols @ g_true))
    if ratio > 1e-10:
        fail.append(f'exact-fit ratio {ratio}')
    # 3. columns orthogonal to the defect -> ratio exactly 1
    base = rng.normal(size=(400, 3))
    d = rng.normal(size=400)
    q, _ = np.linalg.qr(base)
    d_orth = d - q @ (q.T @ d)
    _, ratio, _, _ = solve_bins(base, d_orth)
    if abs(ratio - 1.0) > 1e-9:
        fail.append(f'orthogonal ratio {ratio}')
    # 4. rank-deficient columns do not blow up
    dup = np.concatenate([base, base[:, :1]], axis=1)
    _, ratio, cond, rank = solve_bins(dup, d_orth)
    if rank != 3 or abs(ratio - 1.0) > 1e-9:
        fail.append(f'rank-deficient case rank={rank} ratio={ratio}')
    # 5. nested edges really are nested and monotone
    rmax = 0.0703
    sets = [set(np.round(nested_edges(b, rmax), 12)) for b in (1, 2, 4, 8, 16)]
    for a, b in zip(sets[:-1], sets[1:]):
        if not a <= b:
            fail.append('nested edges are not a refinement hierarchy')
    if tuple(nested_edges(4, rmax)) != PROJECT_EDGES:
        fail.append('B=4 is not FREQUENCY_EDGES')
    for b in (1, 2, 4, 8, 16):
        ed = nested_edges(b, rmax)
        if len(ed) != b + 1 or not np.all(np.diff(ed) > 0):
            fail.append(f'nested edges B={b} malformed: {ed}')
    for b in (4, 16):
        ed = equalwidth_edges(b, rmax)
        if len(ed) != b + 1 or not np.all(np.diff(ed) > 0):
            fail.append(f'equalwidth edges B={b} malformed')
    # 6. masked rfft2 components must sum back to the field, bit-level identity
    field = rng.normal(size=(201, 201))
    spec = np.fft.rfft2(field, norm='ortho')
    rho = np.hypot(np.fft.fftfreq(201, 10.0)[:, None], np.fft.rfftfreq(201, 10.0)[None, :])
    edges = nested_edges(8, float(rho.max()))
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rho >= lo) & (rho < hi)
        parts.append(np.fft.irfft2(np.where(m, spec, 0.0), s=(201, 201), norm='ortho'))
    err = np.abs(np.sum(parts, axis=0) - field).max() / np.abs(field).max()
    if err > 1e-12:
        fail.append(f'band decomposition does not sum back: {err:.3e}')
    for line in fail:
        print('FAIL ' + line)
    print(f'self-test: {"PASS" if not fail else "FAIL"} ({len(fail)} failures)')
    return 1 if fail else 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description='R14 variant-T oracle ceiling (forward only)')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--per-family', type=int, default=4)
    ap.add_argument('--frames-per-band', type=int, default=4)
    ap.add_argument('--require-free-gib', type=float, default=6.0)
    ap.add_argument('--tag', default='anchor29359')
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.device:
        ap.error('--device is required unless --self-test (single shared GPU on this box)')
    if self_test() != 0:
        raise SystemExit('self-test failed; refusing to measure')

    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TREE))
    sys.pycache_prefix = '/tmp/pyc_r14_ceiling'
    import torch

    from grouped_ufno_mionet_v3.config import V3Config
    from grouped_ufno_mionet_v3.data.index import build_manifest
    from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset
    from grouped_ufno_mionet_v3.data.batch import pack_v3_groups
    from grouped_ufno_mionet_v3.evaluation import resolve_onset_frame, ic_window, future_window
    from grouped_ufno_mionet_v3.eval_metrics import FREQUENCY_EDGES
    from grouped_ufno_mionet_v3.model.operator import free_surface_factor
    import scripts.train_grouped_v3_ic8_fullfield_ddp as T

    assert tuple(FREQUENCY_EDGES) == PROJECT_EDGES, (
        f'eval_metrics.FREQUENCY_EDGES changed: {FREQUENCY_EDGES}')

    # ---- GLOBAL calibre: run_identity, never code defaults --------------------
    identity = json.loads(RUN_IDENTITY.read_text())
    eff = identity['effective_config']
    calibre = {k: eff['data'][k] for k in
               ('active_horizon_s', 'dense_time_steps', 'ic_frames', 'time_sampling_policy')}
    # HARD ASSERTION 1: refuse the config.py default that voided two earlier probes.
    assert calibre['active_horizon_s'] != 0.60, (
        'active_horizon_s is the config.py default 0.60; the run effective value is 1.0')
    assert calibre['active_horizon_s'] == 1.0, calibre
    assert calibre['ic_frames'] == 8 and calibre['dense_time_steps'] == 16, calibre

    config = V3Config.from_yaml(PARENT_CFG)
    for key in ('active_horizon_s', 'dense_time_steps', 'ic_frames'):
        built = getattr(config.data, key)
        assert built == calibre[key], (
            f'YAML {key}={built} disagrees with run_identity effective_config '
            f'{calibre[key]}; the model would be built at a calibre the run never used')
    assert eff['model']['dense_radial_cutoff'] == config.model.dense_radial_cutoff

    manifest = build_manifest(config.data.source_h5)
    assert manifest.digest == identity['manifest_digest'], 'manifest digest drift'
    normalizer = T.load_normalizer(config, manifest.digest)
    device = torch.device(args.device)
    if device.type == 'cuda':
        free_b, _ = torch.cuda.mem_get_info(device)
        if free_b / 1024 ** 3 < args.require_free_gib:
            raise SystemExit(f'{device} has {free_b / 1024 ** 3:.2f} GiB free, '
                             f'below --require-free-gib {args.require_free_gib}')

    model = T.build_model(config, dense_checkpoint=True, dense_outer_checkpoint=True)
    model = model.to(device).eval()
    ck = torch.load(ANCHOR, map_location='cpu', weights_only=False)
    incompat = model.load_state_dict(ck['model_state'], strict=False)
    assert set(incompat.missing_keys) == {
        'dense_decoder.s1_lift.weight', 'dense_decoder.s1_lift.bias'}, incompat.missing_keys
    assert not incompat.unexpected_keys, incompat.unexpected_keys
    for name in ('weight', 'bias'):
        assert float(getattr(model.dense_decoder.s1_lift, name).abs().sum()) == 0.0
    scale = float(model.dense_decoder.correction_scale)
    print(f'anchor step {ck.get("global_step")}  correction_scale={scale:.6e}', flush=True)

    def fingerprint():
        h = hashlib.sha256()
        for name, p in sorted(model.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    fp_before = fingerprint()

    records_ds = V3WavefieldDataset(config.data.source_h5, manifest, split='train')
    train_meta = [r for r in manifest.records if r.split == 'train']
    id_to_local = {r.sample_id: i for i, r in enumerate(train_meta)}
    meta_by_id = {r.sample_id: r for r in train_meta}
    excluded = set(json.loads(EXCLUSIONS.read_text())['excluded_group_ids'])

    lists = json.loads(LISTS.read_text())
    picks = []
    for key, family in (('M_list_marmousi', 'marmousi'),
                        ('N_list_layered', 'layered'),
                        ('N_list_uniform', 'uniform')):
        for sid in lists[key][:args.per_family]:
            if sid in id_to_local:
                picks.append((family, sid))
    assert len(picks) == 3 * args.per_family, picks

    axis = np.asarray(manifest.time_s, dtype=np.float64)
    axis_t = torch.tensor(axis, dtype=torch.float64)

    # ---- PER-RECORD calibre: source_t0_s from the record, production windows --
    selection = []
    for family, sid in picks:
        local = id_to_local[sid]
        record = records_ds[local]
        # record.time_s is stored float32, the manifest axis float64; they agree to
        # float32 representation (max observed deviation 2.9e-08).  Anything larger
        # would mean a genuinely different axis.
        assert np.allclose(np.asarray(record.time_s, dtype=np.float64), axis,
                           rtol=0, atol=1e-6), (
            f'{sid} carries a different time axis than the manifest')
        t0 = float(record.source_parameters[3])          # THE per-record calibre
        onset = resolve_onset_frame(axis, t0)
        ic_idx = ic_window(axis, onset, config.data.ic_frames)
        future = future_window(axis, onset, config.data.ic_frames)
        bands = np.array_split(future, 3)
        frames = np.concatenate([
            b[np.linspace(0, len(b) - 1, args.frames_per_band).astype(int)] for b in bands])
        band_of = {int(f): bi for bi, b in enumerate(bands) for f in b}
        horizon_end_t = min(float(axis[-1]), t0 + float(calibre['active_horizon_s']))
        selection.append({
            'family': family, 'sample_id': sid, 'local_index': int(local),
            'group_id': meta_by_id[sid].group_id,
            'in_frozen_exclusion_groups': bool(meta_by_id[sid].group_id in excluded),
            'source_t0_s': t0, 'source_t0_field': 'record.source_parameters[3]',
            'onset': int(onset), 'onset_fn': 'evaluation.resolve_onset_frame',
            'ic_first_frame': int(ic_idx[0]), 'ic_last_frame': int(ic_idx[-1]),
            'future_first_frame': int(future[0]), 'future_last_frame': int(future[-1]),
            'window_fn': 'evaluation.future_window (FULL future window, not horizon-truncated)',
            'active_horizon_end_frame': int(min(np.searchsorted(axis, horizon_end_t),
                                                axis.size - 1)),
            'frames': [int(f) for f in frames],
            'bands': [int(band_of[int(f)]) for f in frames],
        })

    # HARD ASSERTION 2a: t0 pinned to a constant is the historical bug signature.
    onsets = [s['onset'] for s in selection]
    assert len(set(onsets)) > 1, 'all 12 onsets identical: source_t0_s has been pinned'
    assert len(set(round(s['source_t0_s'], 9) for s in selection)) > 1

    # cross-check the frame set against R6 frame for frame
    r6 = json.loads(R6.read_text())
    r6_frames = {}
    for row in r6['rows']:
        r6_frames.setdefault(row['sample_id'], []).append(int(row['frame']))
    for spec in selection:
        assert r6_frames.get(spec['sample_id']) == spec['frames'], (
            f"frame set for {spec['sample_id']} differs from R6: "
            f"{spec['frames']} vs {r6_frames.get(spec['sample_id'])}")
    print(f'{len(selection)} records, {sum(len(s["frames"]) for s in selection)} frames, '
          f'frame set identical to R6', flush=True)

    # ---- radial bin masks ----------------------------------------------------
    grid0 = pack_v3_groups([records_ds[selection[0]['local_index']]])
    # pack_v3_groups returns the shared physical axes as 1-D [n] tensors.
    assert grid0.x_m.ndim == 1 and grid0.z_m.ndim == 1, (grid0.x_m.shape, grid0.z_m.shape)
    x_axis = np.asarray(grid0.x_m.cpu(), dtype=np.float64)
    z_axis = np.asarray(grid0.z_m.cpu(), dtype=np.float64)
    nz, nx = z_axis.size, x_axis.size
    dx, dz = float(x_axis[1] - x_axis[0]), float(z_axis[1] - z_axis[0])
    assert np.allclose(np.diff(x_axis), dx, rtol=1e-12, atol=1e-12)
    assert np.allclose(np.diff(z_axis), dz, rtol=1e-12, atol=1e-12)
    # identical construction to eval_metrics.frequency_metrics
    rho = np.hypot(np.fft.fftfreq(nz, dz)[:, None], np.fft.rfftfreq(nx, dx)[None, :])
    rho_max = float(rho.max())

    schemes = {}
    for b in (1, 2, 4, 8, 16):
        schemes[f'nested_B{b}'] = ('nested', b, nested_edges(b, rho_max))
    for b in (4, 16):
        schemes[f'equalwidth_B{b}'] = ('equalwidth', b, equalwidth_edges(b, rho_max))
    masks, scheme_meta = {}, {}
    for name, (kind, b, edges) in schemes.items():
        mk = [(rho >= lo) & (rho < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        cover = np.sum(mk, axis=0)
        assert cover.min() == 1 and cover.max() == 1, f'{name} masks do not partition rho'
        masks[name] = mk
        scheme_meta[name] = {'kind': kind, 'B': b,
                             'edges_cycles_per_m': [e if np.isfinite(e) else None for e in edges],
                             'modes_per_bin': [int(m.sum()) for m in mk],
                             'empty_bins': int(sum(1 for m in mk if not m.any()))}

    # ---- forward, capturing coarse / y / correction ---------------------------
    decoder = model.dense_decoder
    grab = {'coarse': [], 'y': [], 'corr': []}
    original_forward = decoder.forward

    def wrapped(medium, source, sp, r2m, time_s, coarse, s1_frames=None):
        out = original_forward(medium, source, sp, r2m, time_s, coarse, s1_frames=s1_frames)
        grab['coarse'].append(coarse.detach().reshape(-1, nz, nx))
        grab['y'].append(out.detach().reshape(-1, nz, nx))
        return out

    def out_hook(module, inputs, output):
        grab['corr'].append(output.detach().reshape(-1, nz, nx))

    decoder.forward = wrapped
    handle = decoder.output.register_forward_hook(out_hook)

    rows = []
    recon_max = 0.0
    recipe_dev_max = 0.0
    try:
        with torch.no_grad():
            for spec in selection:
                local = spec['local_index']
                record = records_ds[local]
                frames = np.asarray(spec['frames'])
                macro = pack_v3_groups([record])
                times = axis_t[frames].float().to(device)
                target = records_ds.read_wavefield(local, axis_t[frames].float())
                ic_idx = ic_window(axis, spec['onset'], config.data.ic_frames)
                ic_t = axis_t[ic_idx].float()
                ic_v = records_ds.read_wavefield(local, ic_t).values

                src = macro.source_parameters.to(device)
                assert abs(float(src[0, 3]) - spec['source_t0_s']) < 1e-12
                z_dev = macro.z_m.to(device)
                prepared = model.prepare_sources(
                    model.encode_medium(macro.velocity_mps.to(device), normalizer),
                    src, macro.source_map.to(device), normalizer,
                    record_to_medium=macro.record_to_medium.to(device),
                    ic_snapshots_physical=ic_v[None].to(device),
                    anchor_time_s=torch.tensor([float(ic_t[0])], device=device),
                )
                for key in grab:
                    grab[key].clear()
                prediction = model.dense_normalized(
                    prepared, times, x_m=macro.x_m.to(device), z_m=z_dev, time_block=1)
                assert len(grab['coarse']) == len(frames) == len(grab['corr']), (
                    f'{len(grab["coarse"])} decoder calls / {len(grab["corr"])} head calls '
                    f'for {len(frames)} frames; the capture would be misaligned')
                truth = normalizer.encode_pressure(target.values[None].to(device), src[:, 4])
                coarse = torch.cat(grab['coarse'], 0)
                y_cap = torch.cat(grab['y'], 0)
                corr = torch.cat(grab['corr'], 0)
                surf = free_surface_factor(z_dev)[:, None].expand(nz, nx)

                for i, frame in enumerate(frames):
                    c = coarse[i].double().cpu().numpy()
                    t_ = truth[0, i].double().cpu().numpy()
                    r_exact = corr[i].double().cpu().numpy()
                    r_recipe = ((y_cap[i] - coarse[i]).double() / scale).cpu().numpy()
                    sf = surf.double().cpu().numpy()
                    denom = max(np.abs(r_exact).max(), 1e-300)
                    recipe_dev_max = max(recipe_dev_max,
                                         float(np.abs(r_recipe - r_exact).max() / denom))

                    e_dec = (c - t_).reshape(-1)
                    e_dep = (c * sf - t_).reshape(-1)
                    row = {
                        'family': spec['family'], 'sample_id': spec['sample_id'],
                        'frame': int(frame), 'time_band': spec['bands'][i],
                        'time_s': float(axis[frame]),
                        'defect_norm_decoder': float(np.linalg.norm(e_dec)),
                        'defect_norm_deployed': float(np.linalg.norm(e_dep)),
                        'branch_norm': float(np.linalg.norm(r_exact)),
                        'ratios': {}, 'conds': {}, 'ranks': {}, 'gains': {},
                    }
                    for r_name, r_field in (('exact', r_exact), ('r6recipe', r_recipe)):
                        spec_k = np.fft.rfft2(r_field, norm='ortho')
                        for sname, mk in masks.items():
                            if r_name == 'r6recipe' and sname != 'nested_B1':
                                continue      # the recipe is only needed for the self-check
                            parts = np.stack([
                                np.fft.irfft2(np.where(m, spec_k, 0.0), s=(nz, nx), norm='ortho')
                                for m in mk])
                            recon = float(np.abs(parts.sum(0) - r_field).max()
                                          / max(np.abs(r_field).max(), 1e-300))
                            recon_max = max(recon_max, recon)
                            cols = parts.reshape(len(mk), -1).T          # [P, B]
                            for space, defect, mat in (
                                    ('decoder', e_dec, cols),
                                    ('deployed', e_dep, cols * sf.reshape(-1)[:, None])):
                                g, ratio, cond, rank = solve_bins(mat, defect)
                                key = f'{r_name}|{space}|{sname}'
                                row['ratios'][key] = ratio
                                row['conds'][key] = cond
                                row['ranks'][key] = rank
                                row['gains'][key] = [float(v) for v in g]
                    rows.append(row)
                del prediction, coarse, y_cap, corr, truth
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f'  {spec["family"]}/{spec["sample_id"]}: {len(frames)} frames '
                      f'({time.monotonic() - started:.0f}s)', flush=True)
    finally:
        decoder.forward = original_forward
        handle.remove()

    assert fingerprint() == fp_before, 'parameters changed during a forward-only probe'

    # ---- self-check gates ----------------------------------------------------
    keys = sorted({k for r in rows for k in r['ratios']})
    med = {k: float(np.median([r['ratios'][k] for r in rows])) for k in keys}
    gates = {
        'reconstruction_max_rel': recon_max,
        'reconstruction_ok': bool(recon_max < RECON_TOL),
        'r6_recipe_vs_exact_max_rel': recipe_dev_max,
        'B1_median_decoder_r6recipe': med['r6recipe|decoder|nested_B1'],
        'B1_median_decoder_exact': med['exact|decoder|nested_B1'],
        'r6_published_median': 0.9984024606069797,
        'r6_selfcheck_range': list(R6_SELFCHECK_RANGE),
        'r6_selfcheck_pass': bool(R6_SELFCHECK_RANGE[0]
                                  <= med['r6recipe|decoder|nested_B1']
                                  <= R6_SELFCHECK_RANGE[1]),
    }
    viol = []
    for r in rows:
        for space in ('decoder', 'deployed'):
            seq = [r['ratios'][f'exact|{space}|nested_B{b}'] for b in (1, 2, 4, 8, 16)]
            d = float(np.max(np.diff(seq)))
            if d > MONOTONE_TOL:
                viol.append({'sample_id': r['sample_id'], 'frame': r['frame'],
                             'space': space, 'max_increase': d})
    gates['monotonicity_violations'] = viol
    gates['monotonicity_ok'] = not viol
    gates['weights_unchanged_bitwise'] = True

    # ---- aggregation ---------------------------------------------------------
    def agg(sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return None
        out = {'n': len(sub)}
        for k in keys:
            vals = [r['ratios'][k] for r in sub]
            out[k] = {'median': float(np.median(vals)),
                      'mean': float(np.mean(vals)),
                      'min': float(np.min(vals)),
                      'max': float(np.max(vals))}
        return out

    summary = {'overall': agg(lambda r: True)}
    for family in ('marmousi', 'layered', 'uniform'):
        summary[family] = {'all': agg(lambda r, f=family: r['family'] == f)}
        for band in (0, 1, 2):
            summary[family][f'time_band{band + 1}'] = agg(
                lambda r, f=family, b=band: r['family'] == f and r['time_band'] == b)

    # HARD ASSERTION 2b: identical per-family values are a pinned-t0 signature.
    fam_vals = [summary[f]['all']['exact|deployed|nested_B4']['median']
                for f in ('marmousi', 'layered', 'uniform')]
    gates['per_family_distinct'] = bool(len(set(np.round(fam_vals, 9))) == 3)
    assert gates['per_family_distinct'], (
        f'per-family medians identical ({fam_vals}): the per-record calibre is not varying')

    cond_report = {k: {'min': float(np.min([r['conds'][k] for r in rows])),
                       'median': float(np.median([r['conds'][k] for r in rows])),
                       'max': float(np.max([r['conds'][k] for r in rows])),
                       'rank_deficient_frames': int(sum(
                           1 for r in rows if r['ranks'][k] < scheme_meta[k.split('|')[2]]['B']))}
                   for k in keys}
    gain_report = {}
    for k in keys:
        stacked = np.abs(np.array([r['gains'][k] for r in rows]))
        gain_report[k] = {'median_abs_per_bin': [float(v) for v in np.median(stacked, axis=0)],
                          'median_abs_overall': float(np.median(stacked)),
                          'over_correction_scale':
                              float(np.median(stacked) / scale)}

    payload = {
        'probe': 'R14 variant-T forward oracle ceiling: per-frame free band x time '
                 'diagonal gain on the correction output spectrum',
        'role': 'DIAGNOSTIC, first-order decision aid. NOT a gate judgement, NOT a strict '
                'upper bound on variant T, NOT a falsification.',
        'design': str(OUT / 'CEILING_DESIGN.md'),
        'design_sha256': file_sha256(OUT / 'CEILING_DESIGN.md'),
        'boundary_declaration': [
            'This is the ceiling of a band x time DIAGONAL GAIN family acting on the FINAL '
            'correction output spectrum. The real variant T gate sits at an INTERMEDIATE '
            'layer with further spectral blocks, a 1x1 output conv and nonlinearities after '
            'it, and is per-channel over the retained modes only. The two families are not '
            'nested in either direction: the real gate can do things the output-side family '
            'cannot, and vice versa. First-order decision diagnostic, NOT a strict bound.',
            'DIAGNOSTIC, NOT A GATE JUDGEMENT. Single-record or single-frame ratios must '
            'never be quoted beside family-level gate means (RULED_OUT_ROUTES_20260919.md '
            'section 5.1, same prohibition).',
            'TRAIN split, and the marmousi family contains 50 m translated twin records '
            '(the split leakage disclosed in the paper). The word GENERALISATION must not '
            'be applied to any number in this file.',
            'Every ratio is fitted and evaluated on the same 40401 pixels. With B <= 16 '
            'parameters the in-sample optimism is ~B/P <= 4e-4 in squared terms; it biases '
            'the ceiling DOWNWARDS, so it can only manufacture a false "worth an arm", '
            'never a false "no arm".',
            'g is solved independently per frame, i.e. free in t with no parameterisation, '
            'and absorbs the correction_scale s, so a shut gate cannot depress the number.',
        ],
        'script_sha256': file_sha256(__file__),
        'tree': str(TREE),
        'tree_sha256': {name: file_sha256(TREE / name) for name in (
            'grouped_ufno_mionet_v3/model/dense.py',
            'grouped_ufno_mionet_v3/model/spectral.py',
            'grouped_ufno_mionet_v3/model/operator.py',
            'grouped_ufno_mionet_v3/evaluation.py',
            'grouped_ufno_mionet_v3/eval_metrics.py',
            'scripts/train_grouped_v3_ic8_fullfield_ddp.py') if (TREE / name).is_file()},
        'variant_T_reference': {
            'path': '/root/autodl-tmp/staging/r12_timespec_20260921/time_spectral_gate.py',
            'sha256': file_sha256('/root/autodl-tmp/staging/r12_timespec_20260921/'
                                  'time_spectral_gate.py'),
            'gate': 'g(c, bin(|k|), t) = 1 + A phi(t), A zero-init, phi fixed',
        },
        'checkpoint': {'path': str(ANCHOR), 'sha256': file_sha256(ANCHOR),
                       'global_step': ck.get('global_step'),
                       'correction_scale': scale},
        'config': {'yaml': str(PARENT_CFG), 'yaml_sha256': file_sha256(PARENT_CFG),
                   'digest': config.digest(),
                   'run_identity': str(RUN_IDENTITY),
                   'run_identity_sha256': file_sha256(RUN_IDENTITY),
                   'global_calibre_from_effective_config': calibre,
                   'global_calibre_source': 'run_identity.json -> effective_config.data; '
                                            'config.py defaults are refused '
                                            '(active_horizon_s default 0.60 asserted against)',
                   'per_record_calibre_source': 'record.source_parameters[3] (source_t0_s)',
                   'effective_config': eff},
        'manifest_digest': manifest.digest,
        'grid': {'nz': nz, 'nx': nx, 'dz_m': dz, 'dx_m': dx,
                 'rho_max_cycles_per_m': rho_max,
                 'rho_construction': 'hypot(fftfreq(nz,dz)[:,None], rfftfreq(nx,dx)[None,:]) '
                                     '-- identical to eval_metrics.frequency_metrics'},
        'frequency_edges_project': [e if np.isfinite(e) else None for e in PROJECT_EDGES],
        'bin_schemes': scheme_meta,
        'selection': {'n_records': len(selection), 'n_frames': len(rows),
                      'split': 'train', 'records': selection,
                      'frame_set_identical_to_r6': True,
                      'r6_reference': str(R6), 'r6_sha256': file_sha256(R6)},
        'spaces': {
            'decoder': 'e = coarse - truth, columns = r_b  (R6-compatible, carries the '
                       '0.9984 self-check)',
            'deployed': 'e = coarse*surf - truth, columns = r_b*surf, surf = tanh(z/20)^2 '
                        '(the space the training loss is computed in; carries the verdict)',
        },
        'r_capture': {
            'exact': 'forward hook on dense_decoder.output -> the correction tensor before s',
            'r6recipe': '(y - coarse)/s, the R6 recipe; loses ~1e-2 relative precision to '
                        'float32 cancellation because ||s*r*surf||/||coarse*surf|| ~ 9.3e-06',
        },
        'self_check_gates': gates,
        'median_all_keys': med,
        'summary': summary,
        'condition_numbers': cond_report,
        'fitted_gains': gain_report,
        'rows': rows,
        'wall_s': round(time.monotonic() - started, 1),
    }
    target = OUT / f'CEILING_{args.tag}.json'
    target.write_text(json.dumps(payload, indent=1, allow_nan=False) + '\n')

    print('\n--- self-check gates ---')
    print(f'  B=1 decoder (r6 recipe) median = {gates["B1_median_decoder_r6recipe"]:.6f} '
          f'(R6 published 0.998402) -> {"PASS" if gates["r6_selfcheck_pass"] else "FAIL"}')
    print(f'  sum_b r_b == r          max rel = {recon_max:.3e} -> '
          f'{"PASS" if gates["reconstruction_ok"] else "FAIL"}')
    print(f'  monotone in B                   -> '
          f'{"PASS" if gates["monotonicity_ok"] else "FAIL"} ({len(viol)} violations)')
    print(f'  per-family distinct             -> {gates["per_family_distinct"]}')
    print('\n--- family median best_ratio (exact r) ---')
    for space in ('decoder', 'deployed'):
        for family in ('marmousi', 'layered', 'uniform'):
            line = '  '.join(
                f'B{b}={summary[family]["all"][f"exact|{space}|nested_B{b}"]["median"]:.5f}'
                for b in (1, 2, 4, 8, 16))
            print(f'  {space:9s} {family:9s} {line}')
    print(f'\nwrote {target}  ({payload["wall_s"]}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
