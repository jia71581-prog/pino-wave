#!/usr/bin/env python3
"""GO/NO-GO probe: can the A+1 render backbone's features linearly generate the late
amplitude fields, or is the feature content itself the wall?

Context (VERDICT sec 22-23): late error is not the WKB carrier (single-arrival oracle
already fits late to 0.1-0.3%) and not the synthesis-head rank number per se (adding a
rank-32 late head did not absorb the residual rank). The single-arrival oracle needs
amplitude fields of rank ~38, while A+1 uses rank-8. The open question: is rank-8 merely
a too-small number (widen it and win), or can the render features NOT linearly produce
the rank-38 amplitude structure at all (a feature/backbone wall)?

A+1 builds amplitudes as a_j(x) = sum_r mix[j,r] * basis_r(x), basis_r = Conv1x1(F)_r,
i.e. a per-pixel LINEAR map of the 128-dim render features F(x), rank-limited to R=8. This
probe removes the rank limit: fit the late field with amplitudes = an UNRESTRICTED linear
map of the SAME frozen render features (a_j = W_a[j] @ F(x), W_a in R^{nf x 128}, shared
over pixels), by ridge least-squares. Compare:
  * FEATURE-LINEAR fit relL2 (what the render features can express, any rank <= 128)
  * FREE per-pixel oracle relL2 (sec 23, ~0.1-0.3%)
If feature-linear ~= free oracle -> the features CONTAIN the late structure and the fix is
just more rank in the head (architecture-easy). If feature-linear >> oracle -> the render
features themselves lack the late structure -> must change the backbone.

The WKB carrier and eikonal tau are the same the model uses; only the amplitude source
(frozen render features) is under test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import dataclasses

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.train_saved_time_v4_probe import _model
from scripts.train_grouped_v3_pilot import load_normalizer

NORM = "/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json"


def _capture_render(model, record, normalizer, device):
    """Run the A+1 forward and capture rendered_record [1,width,H,W] via a hook on unet."""
    captured = {}
    lf = model.dense_decoder.local_field if hasattr(model.dense_decoder, "local_field") else None
    # locate the generator that owns .unet + .helmholtz_synthesis
    gen = None
    for m in model.modules():
        if hasattr(m, "helmholtz_synthesis") and getattr(m, "helmholtz_synthesis") is not None and hasattr(m, "unet"):
            gen = m
            break
    if gen is None:
        raise RuntimeError("could not locate the Helmholtz generator with a unet")
    h = gen.unet.register_forward_hook(lambda mod, inp, out: captured.__setitem__("r", out.detach()))
    velocity = record.velocity_mps.to(device).unsqueeze(0)
    source = record.source_parameters.to(device).unsqueeze(0)
    source_map = record.source_map.to(device).unsqueeze(0)
    prepared = model.prepare_sources(
        model.encode_medium(velocity, normalizer), source, source_map, normalizer,
        record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
    )
    dense_grid = model.prepare_dense_grid(
        prepared, x_m=record.x_m.to(device), z_m=record.z_m.to(device),
        travel_time_s=(None if record.dense_travel_time_s is None
                       else record.dense_travel_time_s.to(device).unsqueeze(0)),
    )
    with torch.no_grad():
        model.dense_normalized(prepared, record.time_s.to(device), dense_grid=dense_grid, time_block=1)
    h.remove()
    if "r" not in captured:
        raise RuntimeError("unet hook did not fire (helmholtz path not taken)")
    # the helmholtz render is the LAST unet call with batch==records (1); may capture the
    # per-record render. Take it as [1,width,H,W].
    r = captured["r"]
    if r.shape[0] != 1:
        r = r[:1]
    return r  # [1,width,H,W]


def ridge_linear_fit(features, target_late, tl, tau, omega, device):
    """Fit late field with amplitudes = W @ features (unrestricted linear in features).

    features [C,HW]; target_late [Tl,HW]; returns relL2 of the best feature-linear WKB
    synthesis. The model is p(t,x) = sum_j [ (Wa[j]@F)(x) cos(w_j(t-tau)) + (Wb[j]@F)(x) sin ].
    Stacking over (t): for each pixel this is linear in {Wa,Wb} but Wa,Wb are shared across
    pixels, so we solve one global ridge system in the 2*nf*C unknowns via normal equations
    built by summing over pixels (C=128, nf~48 -> 2*48*128=12288 unknowns; tractable).
    """
    C, HW = features.shape
    nf = omega.shape[0]
    Tl = tl.shape[0]
    # Design per (t,pixel): coefficient of Wa[j,c] is cos(w_j(t-tau_x)) * F[c,x]; similarly sin.
    # Build M [Tl*HW, 2*nf*C] is too big; instead accumulate normal equations A=MtM, y=Mt d.
    # M row (t,x) has entries: cos_jx(t) * F[c,x] for (j,c) in cos block, sin for sin block.
    # We accumulate over pixels in chunks.
    ncol = 2 * nf * C
    A = torch.zeros(ncol, ncol, device=device, dtype=torch.float64)
    y = torch.zeros(ncol, device=device, dtype=torch.float64)
    chunk = 128
    for xs in range(0, HW, chunk):
        xe = min(xs + chunk, HW)
        Fx = features[:, xs:xe].double()                      # [C, c]
        taux = tau[xs:xe].double()                            # [c]
        d = target_late[:, xs:xe].double()                    # [Tl, c]
        arg = omega[None, :, None].double() * (tl[:, None, None].double() - taux[None, None, :])  # [Tl,nf,c]
        cosb = torch.cos(arg); sinb = torch.sin(arg)          # [Tl,nf,c]
        # basis phi[(cos/sin,j), c, t] ; M[(t),(block,j,c)] = phi * F[c]
        # phi_cos [Tl,nf,c] * F[c] -> [Tl,nf,C,c]; flatten (nf,C)
        Mc = (cosb[:, :, None, :] * Fx[None, None, :, :]).reshape(Tl, nf * C, xe - xs)  # [Tl,nfC,c]
        Ms = (sinb[:, :, None, :] * Fx[None, None, :, :]).reshape(Tl, nf * C, xe - xs)
        M = torch.cat([Mc, Ms], dim=1)                        # [Tl, 2nfC, c]
        # accumulate A += sum_x M[:, :, x]^T M[:, :, x]; y += sum_x M[:,:,x]^T d[:,x]
        Mp = M.permute(2, 1, 0)                               # [c, 2nfC, Tl]
        A += torch.einsum("xit,xjt->ij", Mp, Mp)
        y += torch.einsum("xit,tx->i", Mp, d)
    lam = 1e-3 * torch.diagonal(A).mean().clamp_min(1e-12)
    A = A + lam * torch.eye(ncol, device=device, dtype=torch.float64)
    w = torch.linalg.solve(A, y.unsqueeze(-1)).squeeze(-1)    # [2nfC]
    # evaluate relL2
    num = 0.0; den = 0.0
    for xs in range(0, HW, chunk):
        xe = min(xs + chunk, HW)
        Fx = features[:, xs:xe].double(); taux = tau[xs:xe].double(); d = target_late[:, xs:xe].double()
        arg = omega[None, :, None].double() * (tl[:, None, None].double() - taux[None, None, :])
        cosb = torch.cos(arg); sinb = torch.sin(arg)
        Mc = (cosb[:, :, None, :] * Fx[None, None, :, :]).reshape(Tl, nf * C, xe - xs)
        Ms = (sinb[:, :, None, :] * Fx[None, None, :, :]).reshape(Tl, nf * C, xe - xs)
        M = torch.cat([Mc, Ms], dim=1)
        pred = torch.einsum("tix,i->tx", M, w)
        num += float(((pred - d) ** 2).sum()); den += float((d ** 2).sum())
    return (num / max(den, 1e-30)) ** 0.5


def run(*, checkpoint, sample_id, source_index, nf, late_start, stride, device_name, background_cache=None):
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = build_manifest("/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5")
    base = V3Config.from_yaml("configs/grouped_v3/continuous_pilot.yaml")
    base = dataclasses.replace(base, model=dataclasses.replace(base.model, width=128))
    base = dataclasses.replace(base, data=dataclasses.replace(base.data, normalization_json=NORM))
    variant = ProbeVariant(
        depth=8, use_local_phase=True, spectral_rank=112, modes=32, local_field=True,
        local_field_helmholtz_synthesis=True, local_field_helmholtz_synthesis_frequencies=64,
        local_field_helmholtz_synthesis_rank=8,
    )
    model = _model(base, manifest, variant).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"], strict=True)
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)

    ds = GuardedOnsetDataset(
        "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5", manifest,
        split="validation", sample_ids=(sample_id,),
        travel_time_h5="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5",
    )
    record = ds[0]
    feat = _capture_render(model, record, normalizer, device)[0]      # [C,H,W]
    C, H, W = feat.shape
    feat = feat[:, ::stride, ::stride].reshape(C, -1)                 # [C,HW']
    src_amp = record.source_parameters.to(device).unsqueeze(0)[:, 4]
    with h5py.File("/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5", "r", swmr=True) as f:
        truth = torch.tensor(np.asarray(f["wavefield"][source_index], dtype=np.float32), device=device)
        ts = torch.tensor(np.asarray(f["time_s"][:], dtype=np.float64), device=device)
    truth_n = normalizer.encode_pressure(truth.unsqueeze(0), src_amp)[0]  # [T,H,W]
    truth_n = truth_n[:, ::stride, ::stride]
    Hs, Ws = truth_n.shape[-2:]
    with h5py.File("/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5", "r", swmr=True) as f:
        sids = [s.decode() if isinstance(s, bytes) else s for s in f["sample_id"][:]]
        tau = torch.tensor(np.asarray(f["travel_time_s"][sids.index(sample_id)], dtype=np.float32), device=device)
    tau = tau[::stride, ::stride].reshape(-1)
    late = slice(late_start, truth_n.shape[0])
    # CORRECTED target: the synthesis head learns the SCATTERING residual scat = truth - P_bg
    # (A+1's late accuracy comes mostly from the physical background P_bg; the render-fed
    # synthesis only supplies scat). Fitting truth directly asks the features to reproduce
    # P_bg's physics too, which is not their job. With --background-cache we subtract P_bg.
    target_full = truth_n[late].reshape(truth_n[late].shape[0], -1)
    if background_cache:
        from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
        bg = BackgroundFieldProvider(background_cache)
        T_full = truth_n.shape[0]
        pbg = bg.physical([record.sample_id], torch.arange(T_full, device="cpu")[None],
                          device=device, dtype=torch.float32)  # [1,T,H,W] physical full-res
        pbg_n = normalizer.encode_pressure(pbg, src_amp)[0][:, ::stride, ::stride]  # [T,Hs,Ws]
        scat = (truth_n - pbg_n)[late].reshape(truth_n[late].shape[0], -1)
        target_late = scat
        target_kind = "scattering_residual_truth_minus_Pbg"
    else:
        target_late = target_full
        target_kind = "full_field_truth"
    tl = ts[late].float()
    dt = float((ts[-1] - ts[0]) / (len(ts) - 1)); df = 1.0 / (len(ts) * dt)
    omega = (2 * np.pi) * torch.arange(nf, device=device, dtype=torch.float32) * df

    feat_rel = ridge_linear_fit(feat, target_late, tl, tau, omega, device)
    result = {
        "sample_id": sample_id, "nf": nf, "n_features": C, "late_start": late_start,
        "target_kind": target_kind,
        "feature_linear_late_relL2": round(feat_rel, 5),
        "interpretation": "features_sufficient_if_relL2_small (synthesis only needs to fit scat)",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--source-index", type=int, required=True)
    ap.add_argument("--nf", type=int, default=32)
    ap.add_argument("--late-start", type=int, default=201)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--background-cache", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    result = run(checkpoint=args.checkpoint, sample_id=args.sample_id, source_index=args.source_index,
                 nf=args.nf, late_start=args.late_start, stride=args.stride, device_name=args.device,
                 background_cache=args.background_cache)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
