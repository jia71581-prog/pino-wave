"""B2-H checkpointed K-step BACKWARD memory measurement (CODEX addendum requirement).

The addendum is explicit: "one-latent-state arithmetic is not evidence that a
checkpointed K-step backward pass is O(1)."  So MEASURE it.  For increasing rollout
length K, run a full forward+backward and record peak process RSS, with per-step
activation checkpointing ON vs OFF.  Each (K, ckpt) runs in a FRESH subprocess so
resource.getrusage(RUSAGE_SELF).ru_maxrss is a clean per-config high-water mark.

Claim under test: with checkpointing ON, peak backward memory grows SUBLINEARLY in K
(only stored per-step INPUTS, recomputed activations) vs ~LINEAR growth when OFF.
This is a CPU proxy for the GPU activation-memory behavior (same recompute graph).

Run (CPU): CUDA_VISIBLE_DEVICES="" python scripts/measure_b2h_backward_mem.py
"""
from __future__ import annotations

import multiprocessing as mp
import resource
import sys

sys.path.insert(0, "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2")


def _run_one(K, ckpt, Z, X, width, q):
    import torch
    from saved_time_phase_operator_v4.b2h import PhysicalResidualPropagator
    torch.manual_seed(0)
    m = PhysicalResidualPropagator(
        width=width, spectral_rank=width // 2, modes=16, depth=3,
        dt_s=0.0025, dx_m=10.0, dz_m=10.0, gate_init=0.03,
        activation_checkpointing=bool(ckpt),
    ).train()
    b = 1
    p0 = torch.zeros(b, 1, Z, X, requires_grad=False)
    p1 = torch.randn(b, 1, Z, X) * 0.1
    p1.requires_grad_(True)
    vel = torch.full((b, 1, Z, X), 1500.0)
    smap = torch.zeros(b, 1, Z, X); smap[:, :, Z // 2, X // 2] = 1.0
    series = torch.randn(b, K)
    out = m(p0, p1, vel, smap, series, steps=K)
    out.pow(2).mean().backward()
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
    q.put(peak_kb / 1024.0)  # -> MiB


def measure(K, ckpt, Z=128, X=128, width=48):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_one, args=(K, ckpt, Z, X, width, q))
    p.start(); peak = q.get(); p.join()
    return peak


def main():
    Z = X = 128; width = 48
    Ks = [2, 4, 8, 16, 32]
    print(f"B2-H backward peak RSS (MiB), grid {Z}x{X}, width={width}, b=1, CPU float32\n")
    print(f"{'K':>4} | {'ckpt=ON':>10} | {'ckpt=OFF':>10} | {'OFF-ON':>8}")
    print("-" * 44)
    on_vals, off_vals = [], []
    for K in Ks:
        on = measure(K, True, Z, X, width)
        off = measure(K, False, Z, X, width)
        on_vals.append(on); off_vals.append(off)
        print(f"{K:>4} | {on:>10.1f} | {off:>10.1f} | {off - on:>8.1f}")
    # slope of peak vs K (MiB per extra step), late window (K=8->32) to skip fixed base
    def slope(vals):
        i0 = Ks.index(8); i1 = Ks.index(32)
        return (vals[i1] - vals[i0]) / (Ks[i1] - Ks[i0])
    print("\nper-step growth (MiB/step, K=8->32):")
    print(f"  ckpt=ON  : {slope(on_vals):.2f} MiB/step")
    print(f"  ckpt=OFF : {slope(off_vals):.2f} MiB/step")
    print("\nINTERPRETATION: ckpt=ON per-step growth << ckpt=OFF confirms checkpointing")
    print("suppresses stored per-step activations (recompute in backward). Extrapolate ON")
    print("slope to K=401 for the full-trajectory GPU budget (add fixed model+optimizer state).")


if __name__ == "__main__":
    main()
