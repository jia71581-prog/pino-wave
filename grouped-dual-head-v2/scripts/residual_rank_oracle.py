"""Residual separable-rank oracle (CPU diagnostic).

Question this answers: A3's temporal-latent adds a query-independent SEPARABLE
additive correction  d(x,z,t) = sum_k anchor_k(x,z) * trunk_k(t)  on TOP of the
warp_r1 render, with a SHARED (record-independent) temporal basis trunk_k(t).
The best achievable rel-L2 of ANY such rank-r shared-temporal-basis correction on
the warp residual is a strict UPPER BOUND on what A3-warm can reach (the learned,
generalizing basis cannot beat the per-sample SVD optimum). If even this optimistic
floor stays >= 0.08 for a family, the separable FORM is structurally insufficient
for that family -> falsifies A3/A5 rank escalation, not merely the deadlock.

Protocol-correct: reuses the sealed evaluator's exact forward --
  prediction = model.dense_normalized(...)   (normalized space)
  target     = normalizer.encode_pressure(...) (same normalized space)
so residual = target - prediction is EXACTLY what the family_relative_l2 metric sees.

Runs on CPU so it never contends with the GPU training. Diagnostic only -- does not
touch any training artifact.
"""
import argparse, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import _load_context, _load_parent_model


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-per-family", type=int, default=2)
    ap.add_argument("--late-start", type=int, default=160)
    ap.add_argument("--ranks", default="8,16,32,64")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="persist per-record residual/target .npz so a kill is resumable")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all late frames; else cap (smoke)")
    args = ap.parse_args()
    device = torch.device(args.device)
    ranks = [int(r) for r in args.ranks.split(",")]

    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    base, manifest, parent_identity = _load_context(config)
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    run_identity = json.loads((Path(config["artifact_dir"]) / "run" / "run_identity.json").read_text())
    load_checkpoint(args.checkpoint, model=model, map_location=device,
                    expected_manifest_digest=manifest.digest,
                    expected_config_digest=run_identity["run_digest"])
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    dataset = V3WavefieldDataset(base.data.source_h5, manifest, split="validation")

    T = len(manifest.time_s)
    late = list(range(int(args.late_start), T))
    if args.max_frames > 0:
        late = late[: args.max_frames]

    # pick first n records per family
    picked = {}
    for i in range(len(dataset)):
        rec = dataset[i]
        fam = rec.medium_type
        picked.setdefault(fam, [])
        if len(picked[fam]) < args.n_per_family:
            picked[fam].append(i)
    order = [i for fam in sorted(picked) for i in picked[fam]]
    print(f"families={ {k:len(v) for k,v in picked.items()} } late_frames={len(late)} order={order}", flush=True)

    per_family_resid = {}   # fam -> list of (Tlate, HW) residual arrays
    per_family_targ = {}
    sanity = []
    cache = args.cache_dir
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for n, ridx in enumerate(order):
        rec = dataset[ridx]
        fam = rec.medium_type
        cpath = (cache / f"rec{ridx}.npz") if cache is not None else None
        if cpath is not None and cpath.exists():
            z = np.load(cpath)
            R, G = z["R"], z["G"]
            r0 = float(np.linalg.norm(R) / (np.linalg.norm(G) + 1e-12))
            sanity.append((ridx, fam, r0))
            per_family_resid.setdefault(fam, []).append(R)
            per_family_targ.setdefault(fam, []).append(G)
            print(f"[{n+1}/{len(order)}] rec{ridx} {fam} CACHED warp_rel_l2={r0:.4f}", flush=True)
            continue
        velocity = rec.velocity_mps.unsqueeze(0).to(device)
        source = rec.source_parameters.unsqueeze(0).to(device)
        source_map = rec.source_map.unsqueeze(0).to(device)
        prepared = model.prepare_sources(
            model.encode_medium(velocity, normalizer), source, source_map, normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        preds, targs = [], []
        for k in late:
            requested = rec.time_s[k : k + 1]
            tphys = dataset.read_wavefield(ridx, requested)
            times = tphys.requested_time_s.unsqueeze(0).to(device)
            pred = model.dense_normalized(prepared, times, x_m=rec.x_m.to(device),
                                          z_m=rec.z_m.to(device), time_block=1)
            tgt = normalizer.encode_pressure(tphys.values.unsqueeze(0).to(device), source[:, 4])
            preds.append(pred.reshape(-1).float().cpu().numpy())
            targs.append(tgt.reshape(-1).float().cpu().numpy())
        P = np.stack(preds, 0)   # (Tlate, HW)
        G = np.stack(targs, 0)
        R = (G - P).astype(np.float32)
        G = G.astype(np.float32)
        r0 = float(np.linalg.norm(R) / (np.linalg.norm(G) + 1e-12))
        if cpath is not None:
            tmp = cache / f"rec{ridx}.tmp.npz"
            np.savez(tmp, R=R, G=G)
            tmp.replace(cpath)
        sanity.append((ridx, fam, r0))
        per_family_resid.setdefault(fam, []).append(R)
        per_family_targ.setdefault(fam, []).append(G)
        print(f"[{n+1}/{len(order)}] rec{ridx} {fam} warp_rel_l2={r0:.4f} "
              f"elapsed={time.monotonic()-t0:.1f}s", flush=True)

    def tail_floor(stack_R, stack_G, r):
        # shared-temporal-basis rank-r: SVD of (Tlate, Nrec*HW); keep top-r temporal
        # left-singular vectors. remaining rel-L2 vs TARGET energy = comparable to metric.
        M = np.concatenate(stack_R, axis=1)  # (Tlate, Nrec*HW)
        G = np.concatenate(stack_G, axis=1)
        # economy SVD along time
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        r = min(r, len(s))
        Mr = (U[:, :r] * s[:r]) @ Vt[:r]
        remain = np.linalg.norm(M - Mr) / (np.linalg.norm(G) + 1e-12)
        return float(remain)

    def perrec_floor(R, G, r):
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        r = min(r, len(s))
        Rr = (U[:, :r] * s[:r]) @ Vt[:r]
        return float(np.linalg.norm(R - Rr) / (np.linalg.norm(G) + 1e-12))

    def perpixel_freq_floor(R, G, M):
        # A5-PREMISE TEST (dispersive_modal). A5's residual form is a sum of M cosines
        # with a DISTINCT frequency PER PIXEL: d(x,z,t)=sum_m A_m(x,z) cos(w_m(x,z) t + p_m(x,z)).
        # This escapes the separable rank ceiling ONLY IF the late residual's temporal
        # complexity is LOCAL -- different pixels oscillating at different frequencies
        # (multi-arrival / dispersion). Optimistic capability upper bound: per column
        # (pixel) of R (Tlate x HW) keep the top-M DFT temporal bins (each bin = one
        # (freq,amp,phase) mode). If even this per-pixel-free floor stays >= 0.05, A5's
        # per-pixel-frequency form cannot reach the goal -> falsifies A5 BEFORE any GPU.
        # Compared at MATCHED M against shared_temporal_basis_floor[M] (global modes):
        # perpixel << shared  => temporal complexity is local => A5 is the right unlock.
        # perpixel ~= shared  => complexity is global-rank => A5 buys nothing over A3.
        T = R.shape[0]
        F = np.fft.rfft(R, axis=0)              # (Tfreq, HW) complex, HW columns
        if M >= F.shape[0]:
            Fk = F
        else:
            power = np.abs(F)
            idx = np.argpartition(power, -M, axis=0)[-M:]   # (M, HW) top-M bins per pixel
            mask = np.zeros(power.shape, dtype=bool)
            np.put_along_axis(mask, idx, True, axis=0)
            Fk = np.where(mask, F, 0)
        Rr = np.fft.irfft(Fk, n=T, axis=0)      # (Tlate, HW) real reconstruction
        return float(np.linalg.norm(R - Rr) / (np.linalg.norm(G) + 1e-12))

    result = {"late_start": args.late_start, "late_frames": len(late),
              "n_per_family": args.n_per_family, "ranks": ranks,
              "warp_rel_l2_per_record": [{"record": r, "family": f, "rel_l2": v} for r, f, v in sanity],
              "shared_temporal_basis_floor": {}, "perrecord_floor_mean": {},
              "perpixel_freq_floor_mean": {}}
    for fam in sorted(per_family_resid):
        Rs, Gs = per_family_resid[fam], per_family_targ[fam]
        result["shared_temporal_basis_floor"][fam] = {str(r): tail_floor(Rs, Gs, r) for r in ranks}
        pr = {str(r): float(np.mean([perrec_floor(R, G, r) for R, G in zip(Rs, Gs)])) for r in ranks}
        result["perrecord_floor_mean"][fam] = pr
        pf = {str(r): float(np.mean([perpixel_freq_floor(R, G, r) for R, G in zip(Rs, Gs)])) for r in ranks}
        result["perpixel_freq_floor_mean"][fam] = pf
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print("WROTE", args.out, flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
