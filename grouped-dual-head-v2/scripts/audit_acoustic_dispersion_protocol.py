"""Frozen CPU-only local-symbol and metadata audit; never a model promotion gate.

No heterogeneous medium, source startup, free surface, CPML or wavefield truth is
evaluated here. Frequencies above a grid's Nyquist limit are flagged, not assigned
a physical dispersion error. Run --prepare, review the frozen files, then --run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import gammaincinv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fno_acoustic.data_generation.stencils import SECOND_DERIVATIVE_8

NAME = "acoustic_research_revision_20260905"
OUT = ROOT / "results" / NAME
DATA = Path("/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2")
ARMS = [("teacher", 5., .000125), ("r6", 5., .000625),
        ("time_refined", 5., .0003125), ("space_refined", 2.5, .000125),
        ("coarse_125us", 10., .000125), ("coarse_250us", 10., .00025)]
SPEEDS = [1500., 1600., 1800., 3000., 4500., 5800., 6000., 6750.]
F0S = [8., 15., 25., 30.]
TOL = 1.e-10
REPORTS = ["results/r54_scratch_fullpool_40ep_v2_20260829/terminal.json",
           "results/frozen_fine_grid_r6_current_env_reattest_v1_20260905/report.json",
           "results/r6_anchored_r54_device_resident_r1_20260905/runtime_report.json",
           "results/r41_dispersion_transfer_v1_20260828/result.json"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def positive(name, value):
    a = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(name + " must be positive and finite")
    return a


def symbol_1d(theta):
    """-D8 symbol without cos(theta)-1 cancellation; production coefficients."""
    theta = np.asarray(theta, dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must be finite")
    return sum(4. * SECOND_DERIVATIVE_8[4 + j] * np.sin(j * theta / 2.) ** 2
               for j in range(1, 5))


def phase_from_q(q):
    q = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("q must be nonnegative and finite")
    valid = q <= 12.
    x = np.where(valid, q / 4. * (1. - q / 12.), np.nan)
    return 2. * np.arcsin(np.sqrt(np.clip(x, 0., 1.)))


def dispersion(f_hz, c_mps, angle_deg, dx_m, dt_s):
    f = positive("frequency", f_hz)
    c = positive("velocity", c_mps)
    dx = positive("dx", dx_m)
    dt = positive("dt", dt_s)
    angle = np.asarray(angle_deg, dtype=np.float64)
    if not np.all(np.isfinite(angle)) or np.any((angle < 0.) | (angle > 90.)):
        raise ValueError("angle must be finite in [0,90]")
    k = 2. * np.pi * f / c
    tx, tz = k * dx * np.cos(np.deg2rad(angle)), k * dx * np.sin(np.deg2rad(angle))
    kt2 = (symbol_1d(tx) + symbol_1d(tz)) / dx ** 2
    q = dt ** 2 * c ** 2 * kt2
    qmax = dt ** 2 * c ** 2 * 2. * symbol_1d(np.pi) / dx ** 2
    alias = (np.abs(tx) > np.pi) | (np.abs(tz) > np.pi) | (f * dt > .5)
    physical = (~alias) & (q < 6.)
    spatial = np.sqrt(kt2) / k - 1.
    total = phase_from_q(q) / (dt * 2. * np.pi * f) - 1.
    return dict(spatial_error=np.where(physical, spatial, np.nan),
                total_error=np.where(physical, total, np.nan),
                temporal_increment=np.where(physical, total - spatial, np.nan),
                phase_error_1s_rad=np.where(physical, 2. * np.pi * f * total, np.nan),
                ppw=c / (f * dx), q=q, qmax=qmax, mode_spectral_stable=q <= 12.,
                mode_strict_stable=(q > 0.) & (q < 12.), mode_monotone=q < 6.,
                mode_at_double_root=(q == 0.) | (q == 12.),
                full_spectrum_stable=qmax <= 12., full_spectrum_strict=qmax < 12.,
                full_spectrum_safe=qmax <= 9.6, full_spectrum_monotone=qmax < 6.,
                grid_alias=alias, physical=physical)


def restriction_transfer(f_hz, c_mps, angle_deg, input_dx_m=5.):
    k = 2. * np.pi * positive("frequency", f_hz) / positive("velocity", c_mps)
    dx = positive("input dx", input_dx_m)
    angle = np.asarray(angle_deg, dtype=np.float64)
    if not np.all(np.isfinite(angle)) or np.any((angle < 0.) | (angle > 90.)):
        raise ValueError("angle must be finite in [0,90]")
    tx, tz = k * dx * np.cos(np.deg2rad(angle)), k * dx * np.sin(np.deg2rad(angle))
    amp = np.cos(tx / 2.) ** 4 * np.cos(tz / 2.) ** 4
    return dict(amplitude=amp, energy=amp ** 2,
                input_alias=(abs(tx) > np.pi) | (abs(tz) > np.pi),
                output_alias=(abs(2. * tx) > np.pi) | (abs(2. * tz) > np.pi))


def band(f0):
    f0 = float(positive("f0", f0))
    bounds = f0 * np.sqrt(gammaincinv(2.5, np.array([.025, .975])) / 2.)
    f = np.unique(np.r_[np.linspace(*bounds, 257), f0])
    # Trapezoid quadrature, normalized over the retained 95% energy band.
    widths = np.r_[np.diff(f)[0] / 2., (f[2:] - f[:-2]) / 2., np.diff(f)[-1] / 2.]
    w = widths * (f / f0) ** 4 * np.exp(-2. * (f / f0) ** 2)
    return f, w / w.sum(), bounds


def crop_overlap(a, b, width=2000., height=2000.):
    width, height = float(positive("width", width)), float(positive("height", height))
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        raise ValueError("crop origins must be finite pairs")
    return max(0., width - abs(a[0] - b[0])) * max(0., height - abs(a[1] - b[1])) / (width * height)


def metadata_audit(manifest):
    crops, counts = {}, {}
    with Path(manifest).open() as f:
        for line in f:
            row = json.loads(line)
            if row["medium_type"] == "marmousi":
                split = row["split"]
                xy = (float(row["crop_x0_m"]), float(row["crop_z0_m"]))
                crops.setdefault(split, set()).add(xy)
                counts[split] = counts.get(split, 0) + 1
    records = []
    for split in sorted(crops):
        if split == "train":
            continue
        for xy in sorted(crops[split]):
            overlaps = [(crop_overlap(xy, t), t) for t in crops["train"]]
            ov, neighbor = max(overlaps)
            records.append(dict(split=split, x0_m=xy[0], z0_m=xy[1],
                                max_train_overlap_fraction=ov,
                                nearest_train_x0_m=neighbor[0], nearest_train_z0_m=neighbor[1]))
    summary = dict(manifest_sha256=sha(manifest), sample_counts=counts,
                   unique_crop_counts={k: len(v) for k, v in crops.items()},
                   domain_m=[2000., 2000.], metadata_only=True,
                   future_wavefield_accesses=0, h5_accesses=0,
                   interpretation="Crop-area overlap is a geographic dependence diagnostic, not proof of waveform leakage or independent OOD accuracy.")
    for split in crops:
        if split != "train":
            vals = [r["max_train_overlap_fraction"] for r in records if r["split"] == split]
            summary[split] = dict(min_overlap=min(vals), max_overlap=max(vals), mean_overlap=float(np.mean(vals)))
    return summary, records


def production_check():
    import torch
    from fno_acoustic.data_generation.lwc84 import lwc84_step
    from fno_acoustic.data_generation.stencils import laplacian8
    from fno_acoustic.data_generation.restriction import restrict_nodal_2x
    torch.set_num_threads(1)
    z, x = torch.meshgrid(torch.arange(40, dtype=torch.float64),
                          torch.arange(40, dtype=torch.float64), indexing="ij")
    rows = []
    for mx, mz in [(4, 0), (5, 5)]:
        phi = 2. * np.pi * (mx * x + mz * z) / 40.
        field = torch.cos(phi)
        eigen = -float(symbol_1d(2. * np.pi * mx / 40.) + symbol_1d(2. * np.pi * mz / 40.)) / 25.
        observed = laplacian8(field, dx_m=5., dz_m=5., boundary="periodic")
        eig_error = float(torch.max(torch.abs(observed - eigen * field))) / abs(eigen)
        for dt in [.000125, .000625]:
            c = 1500.
            omega_dt = float(phase_from_q(-eigen * c ** 2 * dt ** 2))
            prev, now = field.clone(), torch.cos(phi - omega_dt)
            zero, vel = torch.zeros_like(now), torch.full_like(now, c)
            max_error, phase_error = 0., 0.
            for n in range(2, 65):
                nxt = lwc84_step(p_nm1=prev, p_n=now, q_n=zero, qtt_n=zero,
                                 velocity_mps=vel, dx_m=5., dz_m=5., dt_s=dt, boundary="periodic")
                expected = torch.cos(phi - n * omega_dt)
                max_error = max(max_error, float(torch.max(torch.abs(nxt - expected))))
                coeff = torch.sum(nxt * torch.exp(-1j * phi))
                phase_error = max(phase_error, abs(float(torch.angle(coeff * np.exp(1j * n * omega_dt)))))
                prev, now = now, nxt
            rows.append(dict(mode=[mx, mz], dt_s=dt, eigenvalue=eigen,
                             eigen_relative_linf_error=eig_error, step_linf_error=max_error,
                             phase_error_rad=phase_error, passed=max(eig_error, max_error, phase_error) <= TOL))
    # Odd 41 grid; ignore the two-node reflect boundary halo in the output.
    z, x = torch.meshgrid(torch.arange(41, dtype=torch.float64), torch.arange(41, dtype=torch.float64), indexing="ij")
    tx, tz = .37, .51
    wave = torch.cos(tx * x + tz * z)
    projected = restrict_nodal_2x(wave)
    amp = math.cos(tx / 2.) ** 4 * math.cos(tz / 2.) ** 4
    expected = amp * wave[::2, ::2]
    err = float(torch.max(torch.abs(projected[2:-2, 2:-2] - expected[2:-2, 2:-2])))
    return dict(dtype="float64", device="cpu", grid=[40, 40], steps=64,
                startup_covered=False, initialization="Exact numerical traveling mode at n=0,1; isolates recurrence from startup.",
                tolerance=TOL, modes=rows, restriction_interior_linf_error=err,
                restriction_reflect_boundary_covered=False,
                passed=all(r["passed"] for r in rows) and err <= TOL)


def bindings():
    paths = [Path(__file__), ROOT / "tests/test_acoustic_dispersion_protocol.py",
             DATA / "frozen_config.yaml", DATA / "manifest.jsonl"]
    paths += [ROOT / p for p in REPORTS]
    # Bind every directly imported local package and its numerics dependency.
    paths += sorted((ROOT / "src/fno_acoustic/data_generation").glob("*.py"))
    paths += sorted((ROOT / "src/fno_acoustic/numerics").glob("*.py"))
    paths += [ROOT / "src/fno_acoustic/__init__.py"]
    paths += [ROOT / "saved_time_phase_operator_v4/coarse_lwc84.py",
              ROOT / "scripts/gate_lwc84_cuda_graph_fine_grid_trainonly.py"]
    return {str(p.resolve()): sha(p) for p in paths}


def verify_bindings(expected):
    for path, digest in expected.items():
        if sha(path) != digest:
            raise RuntimeError("Frozen binding changed: " + path)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    prereg = dict(candidate=NAME, schema="local_dispersion_metadata_audit_v1",
        scope="Uniform periodic source-free LWC84 local symbol only; not heterogeneous/source/CPML error floor, nor a neural dispersion-suppression result.",
        arms=[dict(name=n, dx_m=dx, dt_s=dt) for n, dx, dt in ARMS],
        speeds_mps=SPEEDS, f0_hz=F0S, angles_deg=list(range(91)), frequency_points_per_band=257,
        add_f0=True, energy_density="f^4 exp(-2(f/f0)^2), normalized quadrature within 2.5%-97.5% energy quantiles",
        formulas=dict(symbol="4 sum_j a_j sin(j theta/2)^2; production a_j",
                      q="dt^2 c^2 ktilde^2", phase="omega_num dt=2 asin(sqrt(q/4*(1-q/12)))",
                      stability="mode and full spectrum q<=12 reported separately; q=0,12 double-root endpoints are not a strict bounded-recurrence guarantee",
                      monotone="q<6, separate from spectral stability and safety qmax<=9.6",
                      observation="5m->10m binomial5 amplitude=cos(kx*5/2)^4 cos(kz*5/2)^4; energy=amplitude^2; dt_out=0.0025s"),
        observation_policy="Same physical wave on the teacher 5m->10m chain in a separate table. No filter is applied to saved coarse output. Refinement arms are local symbols only; a native 2.5m->10m projection is not evaluated or assumed equivalent.",
        hypotheses=dict(r6_max_abs_local_phase_velocity_error_lte=.001,
                        space_refinement="2.5m vs teacher 5m: max absolute spatial phase-velocity error decreases in every (c,f0) band",
                        time_refinement="312.5us vs R6 625us at 5m: max absolute temporal increment decreases in every (c,f0) band",
                        decision="accept/reject each diagnostic hypothesis, no training, no model promotion"),
        kernel_check=dict(grid=[40, 40], dx_m=5., c_mps=1500., dt_s=[.000125, .000625],
                          modes=[[4, 0], [5, 5]], steps=64, dtype="float64", tolerance=TOL,
                          numeric_mode_initialization=True, startup_covered=False,
                          restriction="production reflect padding; only interior beyond halo compared"),
        metadata_check=dict(domain_m=[2000., 2000.], expected_unique_crops=dict(train=140, validation=30, test_id=30, ood_canonical=1),
                            expected_validation_and_test_overlap=.975, expected_ood_overlap=0.,
                            expected_tolerance=1.e-12, wavefield_accesses=0, checkpoint_loads=0),
        acceptance="Input hashes unchanged; all unit and production checks pass; metadata expectations match; finite interpretable aggregate metrics; CPU budget and disk cap met. Hypothesis rejection is a valid completed diagnostic.",
        failure="Changed binding, unexpected metadata, failed kernel check, H5/checkpoint access, runtime>=180s or total bytes>=50000000 => failed terminal; no claims or promotion.",
        budget=dict(cpu_seconds=180., output_bytes=50_000_000, gpu=False, training=False),
        rollback=dict(action="Preserve existing R6 and R54 entirely; additive diagnostics only; no new checkpoint",
                      r54_best_pt_sha256="2a544eac37f4782193d79f512d55786858c85f93658885bedfc43e9d59a0073a",
                      checkpoint_read_or_loaded=False),
        bindings=bindings(), environment=environment(),
        run_command=f"PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/audit_acoustic_dispersion_protocol.py --run --output {out}")
    write_json(out / "preregistration.json", prereg)
    write_json(out / "freeze.json", dict(preregistration_sha256=sha(out / "preregistration.json"),
                                        script_sha256=sha(__file__), frozen_before_formal_run=True))
    print(json.dumps(dict(preregistration=str(out / "preregistration.json"), freeze=str(out / "freeze.json"))))


def environment():
    import torch
    import scipy
    return dict(python=platform.python_version(), executable=sys.executable, platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, device="cpu")


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate():
    rows, observations, bands = [], [], []
    angles = np.arange(91, dtype=float)[None, :]
    for f0 in F0S:
        freqs, weights, bounds = band(f0)
        f, w = freqs[:, None], weights[:, None] / 91.
        bands.append(dict(f0_hz=f0, energy_low_hz=float(bounds[0]), energy_high_hz=float(bounds[1]), points=len(freqs)))
        for c in SPEEDS:
            teacher = dispersion(f, c, angles, 5., .000125)
            for arm, dx, dt in ARMS:
                d = dispersion(f, c, angles, dx, dt)
                valid = d["physical"]
                def weighted_abs(key):
                    return float(np.sum(np.where(valid, np.abs(d[key]), 0.) * w) / np.sum(valid * w))
                row = dict(arm=arm, dx_m=dx, dt_s=dt, c_mps=c, f0_hz=f0,
                           band_low_hz=float(bounds[0]), band_high_hz=float(bounds[1]),
                           ppw_min=float(np.min(d["ppw"])), q_mode_max=float(np.max(d["q"])), q_full_spectrum=float(d["qmax"]),
                           all_modes_spectral_stable=bool(np.all(d["mode_spectral_stable"])),
                           all_modes_strict_stable=bool(np.all(d["mode_strict_stable"])),
                           all_modes_monotone=bool(np.all(d["mode_monotone"])),
                           full_spectrum_stable=bool(d["full_spectrum_stable"]),
                           full_spectrum_strict=bool(d["full_spectrum_strict"]),
                           full_spectrum_safe=bool(d["full_spectrum_safe"]),
                           full_spectrum_monotone=bool(d["full_spectrum_monotone"]),
                           grid_aliased_modes=int(np.sum(d["grid_alias"])), valid_modes=int(np.sum(valid)),
                           invalid_modes=int(np.sum(~valid)), max_abs_spatial_error=float(np.nanmax(abs(d["spatial_error"]))),
                           min_signed_spatial_error=float(np.nanmin(d["spatial_error"])),
                           max_signed_spatial_error=float(np.nanmax(d["spatial_error"])),
                           max_abs_total_error=float(np.nanmax(abs(d["total_error"]))),
                           min_signed_total_error=float(np.nanmin(d["total_error"])),
                           max_signed_total_error=float(np.nanmax(d["total_error"])),
                           max_abs_temporal_increment=float(np.nanmax(abs(d["temporal_increment"]))),
                           min_signed_temporal_increment=float(np.nanmin(d["temporal_increment"])),
                           max_signed_temporal_increment=float(np.nanmax(d["temporal_increment"])),
                           max_abs_phase_1s_rad=float(np.nanmax(abs(d["phase_error_1s_rad"]))),
                           weighted_abs_spatial_error=weighted_abs("spatial_error"),
                           weighted_abs_total_error=weighted_abs("total_error"),
                           weighted_abs_temporal_increment=weighted_abs("temporal_increment"),
                           weighted_abs_phase_1s_rad=weighted_abs("phase_error_1s_rad"),
                           max_angular_anisotropy=float(np.nanmax(np.nanmax(d["total_error"], axis=1) - np.nanmin(d["total_error"], axis=1))),
                           max_abs_teacher_delta_phase_1s_rad=float(np.nanmax(abs(d["phase_error_1s_rad"] - teacher["phase_error_1s_rad"]))),
                           quadrature_valid_energy_fraction=float(np.sum(valid * w)))
                worst = np.unravel_index(np.nanargmax(abs(d["total_error"])), d["total_error"].shape)
                row.update(worst_total_frequency_hz=float(freqs[worst[0]]),
                           worst_total_angle_deg=float(angles[0, worst[1]]))
                rows.append(row)
            obs = restriction_transfer(f, c, angles)
            observations.append(dict(c_mps=c, f0_hz=f0, chain="teacher_5m_to_10m_only",
                min_amplitude=float(obs["amplitude"].min()), max_amplitude=float(obs["amplitude"].max()),
                energy_weighted_amplitude=float(np.sum(obs["amplitude"] * w)),
                energy_retained_fraction_within_band=float(np.sum(obs["energy"] * w)),
                min_energy_transfer=float(obs["energy"].min()),
                input_aliased_modes=int(obs["input_alias"].sum()), saved_grid_aliased_modes=int(obs["output_alias"].sum()),
                saved_time_aliased_frequencies=int(np.sum(freqs > 1. / (2. * .0025))),
                f0_axis_amplitude=float(restriction_transfer(f0, c, 0.)["amplitude"]),
                f0_axis_energy=float(restriction_transfer(f0, c, 0.)["energy"])))
    return rows, observations, bands


def figures(out, rows, obs):
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm, _, _ in ARMS:
        group = [r for r in rows if r["arm"] == arm and r["c_mps"] == 1500.]
        axes[0].semilogy([r["f0_hz"] for r in group], [100. * r["max_abs_total_error"] for r in group], "o-", label=arm)
    axes[0].axhline(.1, color="black", linestyle="--", linewidth=1, label="R6 hypothesis: 0.1%")
    axes[0].set(xlabel="Ricker f0 (Hz)", ylabel="Max |phase velocity error| (%)", title="Local symbol, c=1500 m/s; 95% energy band")
    axes[0].legend(fontsize=7)
    for c in [1500., 3000., 6000.]:
        group = [r for r in obs if r["c_mps"] == c]
        axes[1].plot([r["f0_hz"] for r in group], [r["energy_retained_fraction_within_band"] for r in group], "o-", label=f"c={c:g} m/s")
    axes[1].set(xlabel="Ricker f0 (Hz)", ylabel="Band energy transmission", ylim=(0, 1.03),
                title="5m to 10m observation filter; not phase correction")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("Homogeneous periodic diagnostics only; no source, startup, boundary or CPML")
    for suffix in ["png", "svg"]:
        path = out / ("dispersion_observation." + suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=160)
    plt.close(fig)


def run(out):
    started = time.monotonic()
    prereg_path = out / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    if sha(prereg_path) != freeze["preregistration_sha256"]:
        raise RuntimeError("Preregistration changed after freeze")
    verify_bindings(prereg["bindings"])
    for name in ["report.json", "terminal.json", "dispersion_summary.csv", "observation_summary.csv", "crop_overlap.csv", "dispersion_observation.png", "dispersion_observation.svg"]:
        if (out / name).exists():
            raise FileExistsError("No overwrite: " + str(out / name))
    ledger = dict(h5_accesses=0, checkpoint_accesses=0, future_wavefield_accesses=0, denied_attempts=0)
    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            suffix = Path(os.fsdecode(args[0])).suffix.lower()
            if suffix in {".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".npy", ".npz"}:
                ledger["denied_attempts"] += 1
                raise RuntimeError("Forbidden data/checkpoint access in CPU metadata audit")
    sys.addaudithook(guard)
    try:
        kernel = production_check()
        if not kernel["passed"]:
            raise RuntimeError("Production kernel check failed")
        import yaml
        config = yaml.safe_load((DATA / "frozen_config.yaml").read_text())
        if not (config["grid"]["dx_m"] == config["grid"]["dz_m"] == 5.
                and config["grid"]["lx_m"] == config["grid"]["lz_m"] == 2000.
                and config["time"]["dt_used_s"] == .000125
                and config["time"]["dt_out_s"] == .0025
                and config["storage_grid"]["dx_m"] == config["storage_grid"]["dz_m"] == 10.
                and config["storage_grid"]["restriction"] == "binomial5_lowpass_then_decimate2"):
            raise RuntimeError("Teacher grid/time/observation chain differs from registered assumptions")
        meta, crops = metadata_audit(DATA / "manifest.jsonl")
        expected = prereg["metadata_check"]["expected_unique_crops"]
        if meta["unique_crop_counts"] != expected:
            raise RuntimeError("Metadata crop counts differ from preregistration")
        for split in ["validation", "test_id"]:
            if max(abs(meta[split]["min_overlap"] - .975), abs(meta[split]["max_overlap"] - .975)) > 1.e-12:
                raise RuntimeError("Heldout crop overlap differs from preregistration")
        if meta["ood_canonical"]["max_overlap"] != 0.:
            raise RuntimeError("OOD overlap differs from preregistration")
        rows, obs, bands = aggregate()
        by = {(r["arm"], r["c_mps"], r["f0_hz"]): r for r in rows}
        r6max = max(r["max_abs_total_error"] for r in rows if r["arm"] == "r6")
        space = all(by["space_refined", c, f]["max_abs_spatial_error"] < by["teacher", c, f]["max_abs_spatial_error"] for c in SPEEDS for f in F0S)
        temporal = all(by["time_refined", c, f]["max_abs_temporal_increment"] < by["r6", c, f]["max_abs_temporal_increment"] for c in SPEEDS for f in F0S)
        decisions = dict(r6_error_lte_0p1percent="accept" if r6max <= .001 else "reject",
                         r6_max_abs_phase_velocity_error=r6max,
                         space_refinement="accept" if space else "reject",
                         time_refinement="accept" if temporal else "reject",
                         promotion=False, target_achieved=False)
        write_csv(out / "dispersion_summary.csv", rows)
        write_csv(out / "observation_summary.csv", obs)
        write_csv(out / "crop_overlap.csv", crops)
        figures(out, rows, obs)
        verify_bindings(prereg["bindings"])
        report = dict(candidate=NAME, preregistration_sha256=sha(prereg_path), code_sha256=sha(__file__),
                      scope=prereg["scope"], decisions=decisions, frequency_bands=bands,
                      production_kernel_check=kernel, metadata_audit=meta, access_ledger=ledger,
                      observations=prereg["observation_policy"], environment=environment(),
                      aggregate_rows=len(rows), observation_rows=len(obs), crop_rows=len(crops),
                      input_hashes_verified_before_and_after=True,
                      physical_frequency_reference="f is continuum frequency, k=2pi f/c. Angle is wavevector direction; errors concern phase velocity, not group velocity.",
                      weighted_metric="Absolute errors integrated with Ricker spectral-energy trapezoid weights over the retained 95% band, uniform angle average. Excludes invalid modes and reports retained quadrature mass.")
        write_json(out / "report.json", report)
        elapsed = time.monotonic() - started
        total_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if elapsed >= 180. or total_bytes >= 50_000_000 or ledger["denied_attempts"]:
            raise RuntimeError("Budget or access gate failed")
        outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        terminal = dict(status="complete", diagnostic_gate_passed=True, model_promoted=False,
                        elapsed_seconds=elapsed, total_output_bytes_before_terminal=total_bytes,
                        output_hashes=outputs, access_ledger=ledger, input_hashes_unchanged=True)
        write_json(out / "terminal.json", terminal)
        print(json.dumps(dict(terminal=str(out / "terminal.json"), decisions=decisions, metadata=meta, elapsed_seconds=elapsed)))
    except Exception as exc:
        if not (out / "terminal.json").exists():
            write_json(out / "terminal.json", dict(status="failed", error=str(exc), model_promoted=False,
                                                   elapsed_seconds=time.monotonic() - started, access_ledger=ledger))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    prepare(args.output.resolve()) if args.prepare else run(args.output.resolve())


if __name__ == "__main__":
    main()
