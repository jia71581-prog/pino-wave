#!/usr/bin/env python3
"""Classical-solver numerical-dispersion comparison on the locked Marmousi
fixed-19-Hz protocol.

Runs two additional classical discretizations of the SAME 2-D acoustic wave
equation on the SAME 201x201 / dx=dz=10 m grid, the same Ricker source and the
same preregistered velocity slices as the archived sealed source-position
study:

  fd2 : second-order space + second-order time explicit finite differences
        (the textbook low-order baseline that exhibits strong grid dispersion)
  ps  : pseudo-spectral (FFT) spatial Laplacian with second-order time stepping,
        the standard quasi dispersion-free anchor (Fornberg 1987)

DFO predictions come verbatim from the archived prediction npz files; the
LWC-84 entry is the archived high-order reference (target_tzx).  No archived
array is modified.

Boundary treatment for BOTH new solvers: top row Dirichlet p=0 (matching the
reference free surface) plus identical cosine-taper damping strips on left /
right / bottom.  All reported metrics are computed inside a fixed interior
analysis window so that boundary treatment cannot drive conclusions.

Outputs per case: snapshot frames (t=0.6 s -> index 240, t=1.0 s -> index 400),
receiver gathers (z=1600 m line), and per-method metrics against the
pseudo-spectral anchor and the archived reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fno_acoustic.data_generation.source import bilinear_point_source

# ---------------------------------------------------------------- constants --
DT_S = 5.0e-4           # internal step for BOTH new solvers; CFL limits are
                        # fd2:  dx/(v_max*sqrt(2)) ~= 1.65 ms,
                        # ps :  2/(v_max*sqrt(2*(pi/dx)^2*2)) ~= 1.05 ms
                        # -> factor >= 2 safety margin either way. Saved frame
                        # times (2.5 ms multiples) align exactly.
FRAME_DT_S = 2.5e-3     # saved frame interval (401 frames over [0, 1] s)
NZ = NX = 201
DX_M = DZ_M = 10.0
SPONGE_W = 30           # cells
SPONGE_K = 14.0         # cosine-damp strength
ANALYSIS = slice(40, 161)   # interior analysis window rows/cols
RECV_Z_IDX = 160            # receiver depth row (z = 1600 m)
RECV_X_STEP = 4             # every 4th column -> x = 40..160 m step 40 m
SNAP_FRAMES = {"t060": 240, "t100": 400}
LATE_START_FRAME = 220      # t >= 0.55 s


def build_sponge(device):
    """Cosine-taper multiplicative mask; 1 in the interior."""
    d = torch.zeros(NZ, NX, dtype=torch.float32)
    ramp = 0.5 * (1.0 + torch.cos(math.pi * torch.arange(SPONGE_W) / SPONGE_W))
    tw = torch.tensor
    for i in range(SPONGE_W):
        w = float(ramp[i])
        # bottom rows
        d[NZ - 1 - i, :] = torch.maximum(d[NZ - 1 - i, :], tw(w))
        # left / right columns (top rows untouched -> no top-side damping)
        d[:, i] = torch.maximum(d[:, i], tw(w))
        d[:, NX - 1 - i] = torch.maximum(d[:, NX - 1 - i], tw(w))
    mask = torch.exp(-(SPONGE_K**2) * d.square())
    return mask.to(device)


def ricker_series(n_steps, f0_hz, t0_s, device):
    t = torch.arange(n_steps, dtype=torch.float32, device=device) * DT_S
    tau = t - t0_s
    a = math.pi ** 2 * f0_hz ** 2
    return (1.0 - 2.0 * a * tau ** 2) * torch.exp(-a * tau ** 2)


def second_derivative_fd(p, dx_m, dz_m):
    """2nd-order central second derivative (interior), zero-flux at rim."""
    out = torch.zeros_like(p)
    out[..., 1:-1, 1:-1] = (
        (p[..., :-2, 1:-1] + p[..., 2:, 1:-1]) / dz_m ** 2
        + (p[..., 1:-1, :-2] + p[..., 1:-1, 2:]) / dx_m ** 2
        - 2.0 * p[..., 1:-1, 1:-1] * (1.0 / dx_m ** 2 + 1.0 / dz_m ** 2)
    )
    return out


def second_derivative_ps(p, dx_m, dz_m):
    """FFT pseudo-spectral Laplacian on the full node-centred box."""
    kx = torch.fft.fftfreq(NX, d=dx_m / (2 * math.pi)).to(p.device)
    kz = torch.fft.fftfreq(NZ, d=dz_m / (2 * math.pi)).to(p.device)
    KX, KZ = torch.meshgrid(kx, kz, indexing="xy")          # [z, x]
    P = torch.fft.fft2(p)
    return torch.real(torch.fft.ifft2(-(KX ** 2 + KZ ** 2) * P))


@torch.no_grad()
def run_solver(velocity, src, f0_hz, t0_s, amp, method, device):
    """Leapfrog simulation returning (frames[401,NZ,NX] saved every 10 steps).

    Wave equation: p_tt = v^2 (L p) + amp*ricker(t)*delta_h   with L in {fd2, ps}.
    """
    vel = torch.as_tensor(velocity, dtype=torch.float32, device=device)
    lap = second_derivative_fd if method == "fd2" else second_derivative_ps

    n_steps = int(round(1.0 / DT_S))            # 4000
    wavelet = ricker_series(n_steps + 2, f0_hz, t0_s, device)
    delta_h = torch.as_tensor(src.delta_h, dtype=torch.float32, device=device)
    force_amp = amp / (DX_M * DZ_M)             # physical point injection

    mask = build_sponge(device)

    p_nm1 = torch.zeros(NZ, NX, device=device)
    p_n = torch.zeros(NZ, NX, device=device)
    frames = torch.empty(401, NZ, NX, dtype=torch.float16, device=device)
    dt2 = DT_S * DT_S

    def store(frame_idx, field):
        frames[frame_idx] = field.half()

    def enforce_boundaries(field):
        field[0, :] = 0.0                          # top free surface
        return field * mask                        # absorbing strips

    save_every = int(round(FRAME_DT_S / DT_S))     # 10
    store(0, p_n)
    prev_acc = None
    for n in range(0, n_steps):
        acc_n = vel.square() * (lap(p_n, DX_M, DZ_M) + delta_h * force_amp * float(wavelet[n]))
        p_np1 = 2.0 * p_n - p_nm1 + dt2 * acc_n
        p_np1 = enforce_boundaries(p_np1)
        if n % save_every == save_every - 1:
            store((n + 1) // save_every, p_np1)
        p_nm1, p_n = p_n, p_np1
    return frames.float()


def rel_l2(pred, ref, eps=1e-12):
    num = float(np.sqrt(np.sum((pred - ref) ** 2)))
    den = max(float(np.sqrt(np.sum(ref ** 2))), eps)
    return num / den


def best_lag_ms(trace, ref_trace, frame_dt=FRAME_DT_S, max_shift_frames=24):
    n = len(ref_trace)
    t = ref_trace - ref_trace.mean()
    best, lag = -2.0, 0
    for s in range(-max_shift_frames, max_shift_frames + 1):
        if s < 0:
            a, b = trace[-s:], ref_trace[: n + s]
        elif s > 0:
            a, b = trace[: n - s], ref_trace[s:]
        else:
            a, b = trace, ref_trace
        c = float(np.corrcoef(a, b)[0, 1]) if len(a) > 8 else 0.0
        if c > best:
            best, lag = c, s
    return lag * frame_dt * 1000.0, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="cap number of cases (0=all)")
    ap.add_argument("--role-prefix", default="interp")
    ap.add_argument("--slice-prefixes", default="", help="comma list like r01,r02 to shard")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "fields").mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    manifest = json.load(open(eval_dir / "prediction_manifest.json"))
    records = manifest["records"]
    allowed = set(filter(None, args.slice_prefixes.split(",")))
    cases = []
    for rec in records:
        cid = rec["record_id"]                    # e.g. marm_r07_interp_center
        suffix = cid.split("_", 2)[-1]
        if not suffix.startswith(args.role_prefix):
            continue
        slice_id = "_".join(cid.split("_")[:2])
        if allowed and slice_id.replace("marm_", "") not in allowed:
            continue
        cases.append({
            "record_id": cid,
            "slice_id": slice_id,
            "role": suffix,
            "prediction": rec["prediction_path"],
        })
    if args.limit:
        cases = cases[: args.limit]
    print(f"[dispersion] {len(cases)} cases, device={device}", flush=True)

    results = []
    t_start = time.time()
    for idx, case in enumerate(cases):
        rid = case["record_id"]
        vel_npz = np.load(eval_dir / "inputs" / f"{case['slice_id']}_velocity.npz")
        velocity = vel_npz["velocity_mps"].astype(np.float32)
        pred_npz = np.load(case["prediction"])
        sp = pred_npz["source_parameters"]
        sx, sz, f0, t0, amp = [float(v) for v in sp]
        ref = np.load(eval_dir / "references" / f"{rid}.npz")["target_tzx"]
        dfo = pred_npz["prediction_tzx"]

        fields = {}
        for method in ("fd2", "ps"):
            fields[method] = run_solver(
                velocity, bilinear_point_source(
                    sx, sz, nx=NX, nz=NZ, dx_m=DX_M, dz_m=DZ_M, centering="node"),
                f0, t0, amp, method, device,
            ).cpu().numpy()

        # receivers along z=1600 m interior columns
        rx_cols = list(range(40, 161, RECV_X_STEP))
        tr_ref = fields["ps"][LATE_START_FRAME:, RECV_Z_IDX, rx_cols].T           # [nrec, nt]
        gathers = {}
        metrics = {"record_id": rid, "slice_id": case["slice_id"], "role": case["role"],
                   "moments": {}}
        for name, F in (("fd2", fields["fd2"]), ("dfo", dfo), ("lwc84", ref)):
            tr = F[LATE_START_FRAME:, RECV_Z_IDX, rx_cols].T
            lags, cors = [], []
            for j in range(tr.shape[0]):
                lag_ms, corr = best_lag_ms(tr[j].astype(np.float64),
                                           tr_ref[j].astype(np.float64))
                lags.append(abs(lag_ms)); cors.append(corr)
            # scale-relative moments so that relative L2 against the pseudo-
            # spectral anchor can be finalized after a global unit calibration
            # (archived fields are stored in the internal pipeline units).
            mom = {}
            for win, sl in (("full", slice(None)), ("late", slice(LATE_START_FRAME, None))):
                X = F[sl, ANALYSIS, ANALYSIS].astype(np.float64)
                Y = fields["ps"][sl, ANALYSIS, ANALYSIS].astype(np.float64)
                mom[win] = {
                    "A": float(np.sum(X * X)),     # sum N^2
                    "B": float(np.sum(X * Y)),     # sum N*ps
                }
                if name == "fd2":
                    mom[win]["C"] = float(np.sum(Y * Y))   # shared sum ps^2
            metrics["moments"][name] = mom
            metrics[name] = {
                "abs_cc_lag_ms_median": float(np.median(lags)),
                "abs_cc_lag_ms_iqr": float(np.subtract(*np.percentile(lags, [75, 25]))),
                "trace_corr_median": float(np.median(cors)),
            }
            if name == "lwc84":
                denom = mom["full"].get("C") or metrics["moments"]["fd2"]["full"]["C"]
                metrics["moments"][name]["full"]["C"] = denom
                metrics["moments"][name]["late"]["C"] = (
                    metrics["moments"]["fd2"]["late"]["C"])
        gathers = {
            "receiver_x_m": (np.asarray(rx_cols) * np.float64(DX_M)),
            "fd2": fields["fd2"][LATE_START_FRAME:, RECV_Z_IDX, rx_cols].T.astype(np.float32),
            "ps": fields["ps"][LATE_START_FRAME:, RECV_Z_IDX, rx_cols].T.astype(np.float32),
            "lwc84": ref[LATE_START_FRAME:, RECV_Z_IDX, rx_cols].T.astype(np.float32),
            "dfo": dfo[LATE_START_FRAME:, RECV_Z_IDX, rx_cols].T.astype(np.float32),
        }
        np.savez_compressed(out_dir / "fields" / f"{rid}.npz", **gathers,
                            snap_t060_fd2=fields["fd2"][240].astype(np.float32),
                            snap_t060_ps=fields["ps"][240].astype(np.float32),
                            snap_t100_fd2=fields["fd2"][400].astype(np.float32),
                            snap_t100_ps=fields["ps"][400].astype(np.float32),
                            snap_t060_lwc84=ref[240].astype(np.float32),
                            snap_t100_lwc84=ref[400].astype(np.float32),
                            snap_t060_dfo=dfo[240].astype(np.float32),
                            snap_t100_dfo=dfo[400].astype(np.float32))
        results.append(metrics)
        if idx % 5 == 0 or idx == len(cases) - 1:
            el = time.time() - t_start
            print(f"  [{idx+1}/{len(cases)}] {rid}  elapsed={el:.0f}s", flush=True)

    # ---- finalize: global unit calibration k-hat -----------------------------
    # Archived fields are stored in internal pipeline units; least squares
    # <lwc84, ps> / <ps, ps> estimates gamma with  arch ~= gamma * physical,
    # i.e. physical = arch / gamma.  All entries below are therefore brought
    # into OUR physical units before forming relative L2 vs the anchor.
    ks = []
    for r in results:
        m = r["moments"]["lwc84"]
        if m["full"]["B"] != 0.0 and m["full"].get("C"):
            ks.append(m["full"]["B"] / m["full"]["C"])     # gamma estimate
    gamma = float(np.median(ks)) if ks else float("nan")
    inv = 1.0 / gamma if ks else 1.0
    print(f"[dispersion] archive->physical scaler = 1/gamma, gamma={gamma:.6e}", flush=True)

    def rel_l2_scaled(A, B, C, alpha):
        """rel L2 between (alpha*N) and ps given moments."""
        num = alpha * alpha * A - 2.0 * alpha * B + C
        return math.sqrt(max(num, 0.0) / C)

    alphas = {"fd2": 1.0, "dfo": inv, "lwc84": inv}
    summary = {}
    for name in ("fd2", "dfo", "lwc84"):
        entries = {
            "abs_cc_lag_ms_median": [r[name]["abs_cc_lag_ms_median"] for r in results],
            "abs_cc_lag_ms_iqr_stat": [r[name]["abs_cc_lag_ms_iqr"] for r in results],
            "trace_corr_median": [r[name]["trace_corr_median"] for r in results],
        }
        for win in ("full", "late"):
            vals = []
            for r in results:
                mom = r["moments"][name][win]
                C = r["moments"]["fd2"][win]["C"]
                vals.append(rel_l2_scaled(mom["A"], mom["B"], C, alphas[name]))
            entries[f"rel_l2_vs_ps_{win}"] = vals
        summary[name] = {
            k: {"median": float(np.median(v)),
                "iqr": float(np.subtract(*np.percentile(v, [75, 25])))}
            for k, v in entries.items()
        }
    payload = {"n_cases": len(results),
               "archive_to_physical_gamma": gamma,
               "summary_vs_pseudo_spectral": summary, "per_case": results}
    with open(out_dir / "dispersion_summary.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
