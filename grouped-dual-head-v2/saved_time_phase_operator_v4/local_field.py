"""Translation-equivariant local-propagation coarse-field generator.

Replaces the global low-rank MIONet coarse field (isolated by the capacity ladder
as the structural floor: it cannot place a sharp, translation-invariant traveling
wavefront without Gibbs ringing) with a spatially-preserving, time-conditioned
conditioned U-Net.  Local convolutions render the ~5-cell wavefront sharply with no
spectral truncation; multi-scale pooling supplies the global receptive field needed
to place the front anywhere.  No global pooling / low-rank product.

Conditioning reuses everything the model already computes: the medium pyramid
(velocity encoder), the 12-channel local-propagation bundle (retarded time, causal
gate, phase, Gabor envelopes), the source map + FiLM(source hidden), and the
saved-time embedding.  A structural causal gate enforces p approximately 0 before
the eikonal first arrival.  Output is one requested saved-time frame per forward, so
arbitrary-saved-time queries stay a hard requirement with no autoregressive rollout.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.source import SourceEncoding
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime
from .spectral import AxisFactorizedComplexSpectralConv2d

from .features import dense_propagation_features


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with the largest group count (<=8) that divides `channels`."""

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class _ConvBlock(nn.Module):
    """Two 3x3 conv + GroupNorm + GELU; batch-size-independent normalization."""

    def __init__(self, in_channels: int, out_channels: int, *, activation_checkpointing: bool, reentrant_checkpoint: bool = False) -> None:
        super().__init__()
        self.checkpoint = bool(activation_checkpointing)
        self.reentrant_checkpoint = bool(reentrant_checkpoint)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_channels)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training and x.requires_grad:
            # Reentrant checkpointing avoids the non-reentrant saved-tensor-hook
            # stack corruption that surfaces when a downstream module (the temporal
            # propagation operator) reads the checkpointed U-Net output — that bug
            # crashes at epoch boundaries after several epochs (pack_hook assert).
            return checkpoint(self._forward, x, use_reentrant=self.reentrant_checkpoint)
        return self._forward(x)


class _UNet(nn.Module):
    """Symmetric conv U-Net with skip connections and odd-size-safe resampling.

    Downsampling uses ceil-mode average pooling (201->101->51->26->13); the decoder
    upsamples each stage back to the exact skip resolution, so odd grid sizes need no
    special padding.  All operations are translation-equivariant convolutions; the
    only spatial mixing across the domain is local conv at successively coarser scales,
    which gives a near-global receptive field while preserving locality.
    """

    def __init__(self, channels: Sequence[int], *, activation_checkpointing: bool, reentrant_checkpoint: bool = False) -> None:
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) < 2:
            raise ValueError("local-field U-Net needs at least two levels")
        self.levels = len(channels)
        _ckpt = dict(activation_checkpointing=activation_checkpointing, reentrant_checkpoint=reentrant_checkpoint)
        self.enc_blocks = nn.ModuleList()
        self.enc_blocks.append(
            _ConvBlock(channels[0], channels[0], **_ckpt)
        )
        for i in range(1, self.levels):
            self.enc_blocks.append(
                _ConvBlock(channels[i - 1], channels[i], **_ckpt)
            )
        self.dec_reduce = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for j in range(self.levels - 1):
            in_c = channels[self.levels - 1 - j]
            out_c = channels[self.levels - 2 - j]
            self.dec_reduce.append(nn.Conv2d(in_c, out_c, kernel_size=1))
            self.dec_blocks.append(
                _ConvBlock(out_c * 2, out_c, **_ckpt)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for i, block in enumerate(self.enc_blocks):
            h = block(h)
            if i < self.levels - 1:
                skips.append(h)
                h = F.avg_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)
        for j, (reduce, block) in enumerate(zip(self.dec_reduce, self.dec_blocks)):
            skip = skips[-(j + 1)]
            h = reduce(h)
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            h = block(torch.cat((h, skip), dim=1))
        return h


class _ContinuousTimePropagationBasis(nn.Module):
    """Query-invariant continuous-time propagation operator.

    The local-field U-Net renders each saved time independently; the wavefront
    position is then re-guessed per frame, producing per-frame position jitter
    (the confirmed position/phase bottleneck).  This module ties the position of
    the wavefront across saved times WITHOUT coupling across queried frames — the
    hard query-invariance contract (single-time queries must match multi-frame
    batches; autoregressive/cross-frame coupling is forbidden).

    It does so with a DeepONet-style factorization (cf. QueryInvariantTemporalBasis):
    a low-rank temporal trunk maps EACH frame's own continuous-time propagation
    features (multi-scale retarded-phase harmonics + travel-progress polynomials)
    to K coefficients, contracted with a spatial coefficient field derived from the
    rendered U-Net features.  Because every frame uses only its own time value and
    the SAME spatial basis, adjacent times share a smooth propagation structure
    (killing jitter) while each frame stays independent of query batching.  A
    zero-initialized gate makes it an exact no-op at warm-start.
    """

    def __init__(
        self, width: int, *, rank: int, harmonics: int = 4, spatial_kernel: int = 1,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if width <= 0 or rank <= 0 or harmonics <= 0:
            raise ValueError("temporal propagation basis width/rank/harmonics must be positive")
        if spatial_kernel <= 0 or spatial_kernel % 2 == 0:
            raise ValueError("temporal propagation basis spatial_kernel must be a positive odd int")
        if gate_init < 0.0:
            raise ValueError("temporal propagation basis gate_init must be >= 0")
        self.rank = int(rank)
        self.harmonics = int(harmonics)
        self.spatial_kernel = int(spatial_kernel)
        # per-frame continuous-time feature dimension:
        #   [1] normalized retarded time  + [2*harmonics] retarded-phase sin/cos
        #   + [3] travel-progress polynomial (linear, quad, cubic)
        feature_dim = 1 + 2 * self.harmonics + 3
        self.time_trunk = nn.Sequential(
            nn.Linear(feature_dim, width), nn.GELU(), nn.Linear(width, self.rank)
        )
        # Spatial basis derived from the rendered U-Net features.  With
        # ``spatial_kernel == 1`` this is a pointwise (1x1) conv: it can only
        # rescale amplitude at each pixel's existing location, NOT translate the
        # wavefront -- which is why a 1x1 operator cannot correct the confirmed
        # position/phase bottleneck.  With ``spatial_kernel > 1`` the basis gains
        # a genuine receptive field (a two-layer k x k stack, RF = 2*k-1), so the
        # gated residual can gather from a spatial neighbourhood and shift energy
        # to fix wavefront position.  The first conv bottlenecks width->rank so the
        # retained intermediate is [R*T, rank, H, W] (memory-frugal: a width-channel
        # intermediate at 201x201 would add ~5 GiB and blow the 24 GiB card); RF and
        # nonlinearity are preserved.  The zero-init gate keeps it an exact warm-start
        # no-op either way; per-frame operation preserves the query-invariance contract.
        if self.spatial_kernel <= 1:
            self.spatial_coefficient = nn.Conv2d(width, self.rank, kernel_size=1)
        else:
            pad = self.spatial_kernel // 2
            self.spatial_coefficient = nn.Sequential(
                nn.Conv2d(width, self.rank, kernel_size=self.spatial_kernel, padding=pad),
                nn.GELU(),
                nn.Conv2d(self.rank, self.rank, kernel_size=self.spatial_kernel, padding=pad),
            )
        # gate_init == 0 -> exact warm-start no-op (byte-reproducible operator ceiling).
        # A small POSITIVE gate_init breaks the A3/B1 cold-start deadlock: at gate==0 the
        # multiplicative residual gives dL/d(time_trunk)=dL/d(spatial_coefficient)=0, so the
        # operator never develops (verified: tempop_warp_b1 gate stalled at -9e-5 through ep8).
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def time_features(
        self, time_s: torch.Tensor, source_parameters: torch.Tensor, *, domain_t_s: float
    ) -> torch.Tensor:
        """Per-frame continuous-time propagation features, shape [records, count, F]."""
        relative = (time_s - source_parameters[:, None, 3]) / float(domain_t_s)  # (R,T)
        frequency = source_parameters[:, None, 2]                                # (R,1)
        feats = [relative]
        for h in range(1, self.harmonics + 1):
            phase = 2.0 * math.pi * frequency * h * (time_s - source_parameters[:, None, 3])
            feats.append(torch.sin(phase))
            feats.append(torch.cos(phase))
        feats.append(relative)                    # linear travel progress
        feats.append(relative * relative)         # quadratic
        feats.append(relative * relative * relative)  # cubic
        return torch.stack(feats, dim=-1)         # (R,T,F)

    def forward(
        self,
        rendered: torch.Tensor,   # (records*count, width, H, W)
        time_s: torch.Tensor,     # (records, count)
        source_parameters: torch.Tensor,
        *,
        records: int,
        count: int,
        domain_t_s: float,
    ) -> torch.Tensor:
        height, width = rendered.shape[-2:]
        feats = self.time_features(time_s, source_parameters, domain_t_s=domain_t_s)  # (R,T,F)
        coeff_t = self.time_trunk(feats)                              # (R,T,K)
        # spatial basis is query-invariant: per (record,frame) rendered features
        # give a K-channel field, contracted with that frame's own time coefficients.
        spatial = self.spatial_coefficient(rendered).reshape(
            records, count, self.rank, height, width
        )                                                            # (R,T,K,H,W)
        residual = torch.einsum("rtk,rtkzx->rtzx", coeff_t, spatial) / math.sqrt(float(self.rank))
        return self.gate * residual                                  # (R,T,H,W)


class _EikonalArrivalWarp(nn.Module):
    """Arrival-aligned characteristic warp of the coarse field (query-invariant).

    The isolated bottleneck is wavefront POSITION/PHASE: the model places the
    moving front slightly off its true location, and a pointwise (or even a small
    receptive-field) amplitude operator cannot fix a *displacement* -- it can only
    reshape amplitude where the energy already is.  Following the transport /
    shift-operator literature (Lanthaler et al., shift-DeepONet, ICLR 2023) and
    eikonal characteristic coordinates (Song et al., PIFNO, 2023), this module
    repositions the field by resampling it along the propagation characteristic:

        u_warp(x, z) = u( (x, z) - s(t,x,z) * d(x,z) ),   d = grad(T)/||grad(T)||

    where ``T(x,z)`` is the eikonal first-arrival time (already available as
    ``travel.seconds``) so ``d`` is the local ray direction, and ``s`` is a bounded
    per-pixel scalar shift (in cell units) predicted from the rendered U-Net
    features.  Moving energy along the ray is exactly the degree of freedom needed
    to correct arrival position/phase.

    Contract preservation:
      * The shift head's final conv is zero-initialised -> ``s == 0`` at warm-start
        -> the sampling grid is the identity -> the warp reproduces its input field
        bit-for-bit (align_corners=True samples exactly at grid nodes).  A
        warm-started checkpoint is therefore an exact no-op until the warp learns.
      * The shift for frame ``i`` depends only on that frame's own rendered features
        plus the static travel-time field, and ``grid_sample`` acts per batch element
        independently, so the per-frame query-invariance contract is preserved (no
        cross-frame coupling, no autoregression).
    """

    def __init__(self, width: int, *, max_shift_cells: float = 8.0, shift_dilation: int = 1) -> None:
        super().__init__()
        if max_shift_cells <= 0:
            raise ValueError("eikonal warp max_shift_cells must be positive")
        dilation = int(shift_dilation)
        if dilation < 1:
            raise ValueError("eikonal warp shift_dilation must be a positive integer")
        self.max_shift_cells = float(max_shift_cells)
        self.shift_dilation = dilation
        hidden = max(width // 2, 1)
        # Two-layer 3x3 head: the shift may depend on a spatial neighbourhood of
        # rendered features, not just a single pixel.  ``shift_dilation`` widens the
        # head's receptive field (RF = 1 + 4*dilation: 5/9/13 for dilation 1/2/3)
        # WITHOUT adding parameters or activation memory -- the 3x3 kernels keep
        # their shape (so a warp checkpoint trained at any dilation loads unchanged)
        # and padding=dilation preserves H,W.  A wider RF lets the head estimate a
        # wavefront displacement from surrounding structure, which single-pixel or
        # RF=5 context cannot resolve for the diagnosed position bottleneck.  Default
        # dilation=1 is byte-identical to the original head.  Final conv zero-init.
        self.shift_head = nn.Sequential(
            nn.Conv2d(width, hidden, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=dilation, dilation=dilation),
        )
        nn.init.zeros_(self.shift_head[-1].weight)
        nn.init.zeros_(self.shift_head[-1].bias)

    @staticmethod
    def _characteristic_direction(arrival: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized grad(T) direction from the arrival field.

        ``arrival`` is ``[records, H, W]``.  Returns ``(d_x, d_z)`` each
        ``[records, H, W]`` -- the unit ray direction along width (x) and height (z).
        Where the travel-time gradient vanishes the eps floor makes ``d`` ~ 0, so no
        spurious shift is applied in flat regions.
        """
        grad_z, grad_x = torch.gradient(arrival, dim=(-2, -1))
        norm = torch.sqrt(grad_x * grad_x + grad_z * grad_z) + 1e-6
        return grad_x / norm, grad_z / norm

    def forward(
        self,
        field: torch.Tensor,     # (records, count, H, W)
        rendered: torch.Tensor,  # (records*count, width, H, W)
        arrival: torch.Tensor,   # (records, H, W)
    ) -> torch.Tensor:
        records, count, height, width = field.shape
        shift = self.max_shift_cells * torch.tanh(self.shift_head(rendered))  # (R*T,1,H,W)
        shift = shift.reshape(records, count, height, width)
        d_x, d_z = self._characteristic_direction(arrival)                    # (R,H,W)
        d_x = d_x[:, None]                                                    # (R,1,H,W)
        d_z = d_z[:, None]
        # identity sampling grid in normalized [-1, 1] coords, layout (x, y)=(W, H)
        device = field.device
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=field.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=field.dtype)
        base_z, base_x = torch.meshgrid(ys, xs, indexing="ij")               # (H,W)
        # sample coords = identity - s * d, with the cell-unit shift converted to
        # normalized units (2 spans the full [-1,1] extent over N-1 cells).
        sample_x = base_x[None, None] - shift * d_x * (2.0 / max(width - 1, 1))
        sample_z = base_z[None, None] - shift * d_z * (2.0 / max(height - 1, 1))
        grid = torch.stack((sample_x, sample_z), dim=-1).reshape(
            records * count, height, width, 2
        )
        warped = F.grid_sample(
            field.reshape(records * count, 1, height, width),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.reshape(records, count, height, width)


class _MultiArrivalWarp(nn.Module):
    """A4 (class-A): extra-arrival mixture of characteristic warps.

    An ADDITIVE later-arrival module applied ON TOP of the r4 ``_EikonalArrivalWarp``
    (``self.warp``), targeting the DOMINANT layered+marmousi *late* residual
    (co-dominant ~79% of aggregate): late frames carry reflections / multiples that
    a single first-arrival warp cannot place -- they are the same wavefront geometry
    re-arriving *later* and slightly *displaced*.

    IMPORTANT (Option B warm-start): this module does NOT own the primary
    first-arrival path.  Path 0 stays the parent-trained ``self.warp``, whose weights
    load unchanged from the warp ceiling checkpoint.  ``num_paths`` here is the number
    of EXTRA delayed paths ``k = 0..num_paths-1``.  Each extra path:
      * reuses the shared first-arrival characteristic ``grad(T)`` (a reflection
        retraces the same rays),
      * is displaced by its own zero-init bounded shift head (transport, not rescale),
      * is delayed by a learned non-negative per-record delay map ``Delta_k(spatial)``
        (query-independent) so it turns on only after ``arrival + Delta_k``,
      * is temporally gated by its own causal sigmoid at that delayed onset,
      * is mixed by a zero-init scalar gate ``g_k`` (an EXACT warm-start no-op: extra
        paths contribute nothing until ``g_k`` learns off zero).

        ``field_out = field_in + sum_{k} g_k * causal_k * warp_k(field_in)``

    where ``field_in`` is the field ALREADY warped by ``self.warp`` (path 0).  At
    initialisation every ``g_k == 0`` so ``field_out == field_in`` EXACTLY -- the
    warm-started model reproduces the loaded warp ceiling byte-for-byte, and only the
    fresh ``multi_arrival.*`` parameters develop (own LR group).  Cost is ``K`` small
    shift-head convs + ``K`` grid-samples on the already-rendered field plus ``K`` 1x1
    delay convs -- no extra U-Net render, same activation-memory envelope as r4.
    """

    def __init__(
        self,
        width: int,
        *,
        num_paths: int = 3,
        max_shift_cells: float = 8.0,
        max_delay_frac: float = 0.5,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if max_shift_cells <= 0:
            raise ValueError("multi-arrival warp max_shift_cells must be positive")
        if num_paths < 1:
            raise ValueError("multi-arrival warp num_paths must be >= 1")
        if not 0.0 < max_delay_frac <= 1.0:
            raise ValueError("multi-arrival warp max_delay_frac must be in (0, 1]")
        if gate_init < 0.0:
            raise ValueError("multi-arrival warp gate_init must be >= 0")
        # num_paths == number of EXTRA delayed paths (path 0 is the external self.warp).
        self.num_paths = int(num_paths)
        self.max_shift_cells = float(max_shift_cells)
        self.max_delay_frac = float(max_delay_frac)
        hidden = max(width // 2, 1)

        def _shift_head() -> nn.Sequential:
            head = nn.Sequential(
                nn.Conv2d(width, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            return head

        # One shift head + one delay head per EXTRA path.
        self.shift_heads = nn.ModuleList(_shift_head() for _ in range(self.num_paths))
        # Delay heads: 1x1 conv on the query-independent record conditioning ->
        # a per-record delay map (query invariant).
        self.delay_heads = nn.ModuleList(
            nn.Conv2d(width, 1, kernel_size=1) for _ in range(self.num_paths)
        )
        # Scalar mixing gate per extra path.  gate_init == 0 -> exact warm-start
        # no-op (byte-reproducible warp ceiling).  A small POSITIVE gate_init breaks
        # the zero-init cold-start deadlock: with a frozen parent, a zero gate gives
        # the shift/delay heads ~zero gradient (dL/d(head) = gate * ... ), so the
        # paths can never develop; a nonzero gate lets both the gate AND the heads
        # receive real gradient (optimizer is then free to grow or shrink the gate).
        self.path_gate = nn.Parameter(torch.full((self.num_paths,), float(gate_init)))

    @staticmethod
    def _characteristic_direction(arrival: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad_z, grad_x = torch.gradient(arrival, dim=(-2, -1))
        norm = torch.sqrt(grad_x * grad_x + grad_z * grad_z) + 1e-6
        return grad_x / norm, grad_z / norm

    def _warp_once(
        self,
        field: torch.Tensor,       # (R, T, H, W)
        shift_head: nn.Module,
        rendered: torch.Tensor,    # (R*T, width, H, W)
        d_x: torch.Tensor,         # (R, 1, H, W)
        d_z: torch.Tensor,         # (R, 1, H, W)
        base_x: torch.Tensor,      # (H, W)
        base_z: torch.Tensor,      # (H, W)
    ) -> torch.Tensor:
        records, count, height, width = field.shape
        shift = self.max_shift_cells * torch.tanh(shift_head(rendered))  # (R*T,1,H,W)
        shift = shift.reshape(records, count, height, width)
        sample_x = base_x[None, None] - shift * d_x * (2.0 / max(width - 1, 1))
        sample_z = base_z[None, None] - shift * d_z * (2.0 / max(height - 1, 1))
        grid = torch.stack((sample_x, sample_z), dim=-1).reshape(
            records * count, height, width, 2
        )
        warped = F.grid_sample(
            field.reshape(records * count, 1, height, width),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.reshape(records, count, height, width)

    def forward(
        self,
        field: torch.Tensor,          # (R, T, H, W) -- already warped by path-0 self.warp
        rendered: torch.Tensor,       # (R*T, width, H, W)
        arrival: torch.Tensor,        # (R, H, W) first-arrival travel time (seconds)
        spatial: torch.Tensor,        # (R, width, H, W) query-independent conditioning
        time_s: torch.Tensor,         # (R, T) absolute saved times (seconds)
        onset: torch.Tensor,          # (R, 1, 1, 1) source onset time (seconds)
        causal_width_s: float,        # causal sigmoid width (seconds)
        domain_t_s: float,            # domain time span (seconds), scales max delay
    ) -> torch.Tensor:
        records, count, height, width = field.shape
        d_x, d_z = self._characteristic_direction(arrival)  # (R,H,W)
        d_x = d_x[:, None]                                  # (R,1,H,W)
        d_z = d_z[:, None]
        device = field.device
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=field.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=field.dtype)
        base_z, base_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

        # ``field`` is already the path-0 (self.warp) first-arrival-warped field.
        # Add each extra delayed/displaced later arrival on top; every gate is
        # zero-init so this returns ``field`` byte-for-byte at warm-start.
        out = field
        max_delay_s = self.max_delay_frac * float(domain_t_s)
        for k in range(self.num_paths):
            warped_k = self._warp_once(
                field, self.shift_heads[k], rendered, d_x, d_z, base_x, base_z
            )
            # Learned non-negative per-record delay map (query-independent).
            delay_k = max_delay_s * torch.sigmoid(
                self.delay_heads[k](spatial)
            ).reshape(records, height, width)                          # (R,H,W)
            # Causal onset of the delayed arrival: appears after arrival + Delta_k.
            tau_k = (
                time_s[:, :, None, None]
                - onset
                - arrival[:, None]
                - delay_k[:, None]
            )                                                          # (R,T,H,W)
            causal_k = torch.sigmoid(tau_k / causal_width_s)
            out = out + self.path_gate[k] * causal_k * warped_k
        return out


class _DynamicGreenScatteringKernel(nn.Module):
    """Dynamic Green/scattering kernel whose support radius follows travel time.

    After the eikonal warp aligns wavefront POSITION, the residual late-time error is
    scattered / reverberated energy: a propagating pulse deposits a wake whose spatial
    support GROWS with elapsed travel time (the Green's-function support of the wave
    operator expands as ``t`` increases).  A fixed pointwise or single-radius operator
    cannot represent this scale-growing scattering -- exactly the r5 escalation the
    warp+amplitude stack (B1) leaves on the table if it only smooths amplitude.

    Following GreenONet (Aldirany et al. 2023) and the background+scattered-field
    decomposition (Ma & Alkhalifah 2025), this module learns a compact bank of
    depthwise spatial stencils at increasing dilations (radii); per-pixel, per-frame
    mixing weights select which radius dominates, and the mixing logits are biased by
    the normalized elapsed travel-progress so the EFFECTIVE support radius follows
    travel time.  The scattered field is added as a zero-init-gated residual on top of
    the strong (warped) parent field -- the parent is never regenerated.

    Contract preservation:
      * ``gate`` is a zero-init scalar -> exact no-op at warm-start (the returned
        residual is identically zero, so a warm-started checkpoint reproduces its
        pre-kernel field bit-for-bit until the kernel learns).
      * frame ``i`` uses only its own warped field, its own rendered features and its
        own elapsed-travel-progress; the depthwise convs and softmax act per
        ``(record, frame)`` sample independently, so the per-frame query-invariance
        contract is preserved (no cross-frame coupling, no autoregression).
    """

    def __init__(
        self, width: int, *, kernel_size: int = 5, dilations: Sequence[int] = (1, 2, 4)
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("green scattering kernel width must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("green scattering kernel_size must be a positive odd int")
        dils = tuple(int(d) for d in dilations)
        if len(dils) == 0 or any(d <= 0 for d in dils):
            raise ValueError("green scattering dilations must be a non-empty list of positive ints")
        self.kernel_size = int(kernel_size)
        self.dilations = dils
        n = len(dils)
        pad = self.kernel_size // 2
        # One depthwise scattering stencil per radius, applied to the single-channel
        # (warped) field.  These are the learnable Green/scattering bases; the zero-init
        # gate -- not the stencil weights -- guarantees the warm-start no-op, so the
        # stencils are free to learn meaningful scattering kernels.
        self.stencils = nn.ModuleList(
            [
                nn.Conv2d(
                    1, 1, kernel_size=self.kernel_size,
                    padding=pad * d, dilation=d, bias=False,
                )
                for d in dils
            ]
        )
        # Per-pixel radius-mixing logits from the rendered U-Net features (1x1: the
        # choice of which scattering radius applies is a local, query-invariant map).
        self.mix = nn.Conv2d(width, n, kernel_size=1)
        # Learnable per-radius bias on the elapsed-travel-progress term: a positive
        # bias on a larger-dilation branch makes late-time pixels weight the larger
        # support, so the effective scattering radius grows with travel time.
        self.radius_bias = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(
        self,
        field: torch.Tensor,      # (records, count, H, W)  -- the warped parent field
        rendered: torch.Tensor,   # (records*count, width, H, W)
        progress: torch.Tensor,   # (records, count, H, W)  -- normalized elapsed progress >= 0
        *,
        records: int,
        count: int,
    ) -> torch.Tensor:
        height, width = field.shape[-2:]
        flat = field.reshape(records * count, 1, height, width)
        branches = torch.cat([stencil(flat) for stencil in self.stencils], dim=1)  # (R*T,n,H,W)
        logits = self.mix(rendered)                                                # (R*T,n,H,W)
        prog = progress.reshape(records * count, 1, height, width)
        logits = logits + self.radius_bias.view(1, -1, 1, 1) * prog
        weights = torch.softmax(logits, dim=1)                                     # (R*T,n,H,W)
        scattered = (weights * branches).sum(dim=1, keepdim=True)                  # (R*T,1,H,W)
        return self.gate * scattered.reshape(records, count, height, width)


class _ContinuousTemporalLatentBasis(nn.Module):
    """Query-invariant continuous temporal LATENT BASIS (candidate A3, class-A).

    This is a COARSE-FIELD (class-A) enrichment, distinct from
    ``_ContinuousTimePropagationBasis`` (a class-B corrector whose K spatial
    channels are read from the *per-frame, time-conditioned* U-Net render, so it
    only reshapes each frame's own render).  A3 instead builds a bank of M
    **query-independent** spatial anchor maps ``L_m(x,z)`` from the record-level,
    time-INDEPENDENT conditioning (medium pyramid + source), SHARED across every
    saved time of a record, and combines them with continuous coefficients
    ``w_m(t, tau)`` from each frame's own scalar-time features:

        residual(x,z,t) = gate * sum_m w_m(t, tau) * L_m(x,z) / sqrt(M).

    Because the anchors do not depend on ``t`` and each frame uses only its own
    ``(t, tau)``, the same physical time yields the same output regardless of which
    other frames share the batch -> per-frame query invariance holds BY
    CONSTRUCTION (no queried-index-axis mixing; this resolves A1 blocker-1).  The
    anchors are computed ONCE per record (not per frame) and there is no K-frame
    U-Net re-render, so activation cost is ~M extra channels per record, not K
    extra renders (this resolves A1 blocker-2, the memory obstacle).  A zero-init
    gate makes it an exact warm-start no-op.

    Rationale (RE-VERIFIED 2026-07-30; the earlier "uniform-floor ceiling bound"
    rationale was RETRACTED): the strict aggregate is the equal-RECORD-weighted mean
    of per-record rel-L2 (streaming_metrics.py:227), and the 48-record panel splits
    LAYERED 19 / MARMOUSI 19 / uniform 10 (integer-verified), so aggregate is
    CO-DOMINATED by layered+marmousi (~79% together), NOT gated by a uniform floor
    (~21%).  A3 is therefore a LEGITIMATE PARALLEL class-A option, not "the only
    goal-critical path": it enriches the coarse field with a continuous low-rank
    temporal basis so a fixed (x,z) can encode multi-arrival/late-time structure a
    single render cannot -- and to matter it must move the DOMINANT layered/marmousi
    late error, which is the pre-registered A3 gate.
    """

    def __init__(self, width: int, *, rank: int, harmonics: int = 4, gate_init: float = 0.0) -> None:
        super().__init__()
        if width <= 0 or rank <= 0 or harmonics <= 0:
            raise ValueError("temporal latent basis width/rank/harmonics must be positive")
        if gate_init < 0.0:
            raise ValueError("temporal latent basis gate_init must be >= 0")
        self.rank = int(rank)
        self.harmonics = int(harmonics)
        # query-independent spatial anchor bank from record-level conditioning.
        # A small 3x3 -> GELU -> 3x3 stack (RF=5) gives the anchors spatial
        # structure; the intermediate is only [records, rank, H, W] (per-record,
        # NOT per-frame) so it is memory-frugal.
        self.anchor_projection = nn.Sequential(
            nn.Conv2d(width, self.rank, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.rank, self.rank, kernel_size=3, padding=1),
        )
        # per-frame continuous-time feature dimension mirrors the propagation basis:
        #   [1] normalized retarded time + [2*harmonics] retarded-phase sin/cos
        #   + [3] travel-progress polynomial (linear, quad, cubic).
        feature_dim = 1 + 2 * self.harmonics + 3
        self.time_trunk = nn.Sequential(
            nn.Linear(feature_dim, width), nn.GELU(), nn.Linear(width, self.rank)
        )
        # gate_init == 0 -> exact warm-start no-op.  A small positive value breaks the
        # cold-start deadlock (see _MultiArrivalWarp): a zero gate gives the anchor/time
        # trunks ~zero gradient on a frozen parent, so the basis can never align.
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def time_features(
        self, time_s: torch.Tensor, source_parameters: torch.Tensor, *, domain_t_s: float
    ) -> torch.Tensor:
        """Per-frame continuous-time features, shape [records, count, F]."""
        relative = (time_s - source_parameters[:, None, 3]) / float(domain_t_s)  # (R,T)
        frequency = source_parameters[:, None, 2]                                # (R,1)
        feats = [relative]
        for h in range(1, self.harmonics + 1):
            phase = 2.0 * math.pi * frequency * h * (time_s - source_parameters[:, None, 3])
            feats.append(torch.sin(phase))
            feats.append(torch.cos(phase))
        feats.append(relative)                        # linear travel progress
        feats.append(relative * relative)             # quadratic
        feats.append(relative * relative * relative)  # cubic
        return torch.stack(feats, dim=-1)             # (R,T,F)

    def forward(
        self,
        conditioning: torch.Tensor,   # (records, width, H, W) -- time-INDEPENDENT
        time_s: torch.Tensor,         # (records, count)
        source_parameters: torch.Tensor,
        *,
        domain_t_s: float,
    ) -> torch.Tensor:
        records, count = time_s.shape
        height, width = conditioning.shape[-2:]
        anchors = self.anchor_projection(conditioning)                # (R,M,H,W) query-independent
        feats = self.time_features(time_s, source_parameters, domain_t_s=domain_t_s)  # (R,T,F)
        coeff_t = self.time_trunk(feats)                              # (R,T,M)
        residual = torch.einsum("rtm,rmzx->rtzx", coeff_t, anchors) / math.sqrt(float(self.rank))
        return self.gate * residual                                  # (R,T,H,W)


class _DispersiveModalField(nn.Module):
    """Query-invariant per-pixel LEARNED-DISPERSION modal coarse field (candidate A5).

    A5 is the class-A ESCALATION the A3 falsification gate points to.  Where A3
    (``_ContinuousTemporalLatentBasis``) modulates query-independent anchors with a
    GENERIC fixed-frequency harmonic trunk, A5 gives each location its OWN oscillatory
    time law with a LEARNED per-pixel angular frequency:

        residual(x,z,t) = gate * (1/sqrt(M)) *
            sum_m A_m(x,z) * cos( omega_m(x,z) * t_rel + phi_m(x,z) ),

    where ``t_rel = (t - onset)/domain_t_s`` is the per-frame normalized retarded time
    and ``A_m, omega_m, phi_m`` are M query-INDEPENDENT maps produced ONCE per record
    from the time-independent conditioning (medium pyramid + source), each via a
    3x3 -> GELU -> 3x3 stack (RF=5).  ``omega_m = 2*pi*max_frequency*sigmoid(raw)`` is
    bounded to (0, max_frequency) cycles over the normalized window so the modes cannot
    diverge.  ``-omega_m*tau`` per-pixel arrival delays are absorbed into ``phi_m``.

    WHY (over A3): multi-arrival + late reverberation in layered/marmousi media is a
    multi-MODE, location-dependent oscillation (a discrete dispersive / Green-modal
    expansion); a single time-conditioned U-Net render or a fixed-frequency harmonic
    bank cannot encode location-varying dispersion.  A5 makes the per-mode frequency a
    learned function of (x,z), which is the single structural factor A3 lacks.

    Query invariance holds BY CONSTRUCTION: each frame's output depends only on its own
    ``t_rel`` and the query-independent maps (no queried-index-axis mixing).  The maps
    are computed once per record; the cos-combine is done in a FRAME-CHUNKED loop so the
    (chunk, M, H, W) intermediate stays bounded regardless of ``count`` -- so the full
    401-frame validation panel cannot blow up memory the way an un-chunked (R,T,M,H,W)
    tensor would.  ``gate_init == 0`` is an exact warm-start no-op; launch with a small
    positive gate_init to avoid the A3 cold-start deadlock.
    """

    def __init__(
        self,
        width: int,
        *,
        modes: int,
        max_frequency: float = 8.0,
        gate_init: float = 0.0,
        frame_chunk: int = 64,
    ) -> None:
        super().__init__()
        if width <= 0 or modes <= 0:
            raise ValueError("dispersive modal field width/modes must be positive")
        if max_frequency <= 0.0:
            raise ValueError("dispersive modal field max_frequency must be > 0")
        if gate_init < 0.0:
            raise ValueError("dispersive modal field gate_init must be >= 0")
        if frame_chunk <= 0:
            raise ValueError("dispersive modal field frame_chunk must be positive")
        self.modes = int(modes)
        self.max_frequency = float(max_frequency)
        self.frame_chunk = int(frame_chunk)

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(width, self.modes, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.modes, self.modes, kernel_size=3, padding=1),
            )

        self.amp_projection = _head()    # A_m(x,z)
        self.omega_projection = _head()  # raw -> bounded omega_m(x,z)
        self.phi_projection = _head()    # phi_m(x,z) (absorbs per-pixel arrival delay)
        # gate_init == 0 -> exact warm-start no-op.  A small positive value breaks the
        # cold-start deadlock (see _ContinuousTemporalLatentBasis / _MultiArrivalWarp).
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def forward(
        self,
        conditioning: torch.Tensor,   # (records, width, H, W) -- time-INDEPENDENT
        time_s: torch.Tensor,         # (records, count)
        source_parameters: torch.Tensor,
        *,
        domain_t_s: float,
    ) -> torch.Tensor:
        records, count = time_s.shape
        height, w = conditioning.shape[-2:]
        amp = self.amp_projection(conditioning)                       # (R,M,H,W)
        omega = (2.0 * math.pi * self.max_frequency) * torch.sigmoid(
            self.omega_projection(conditioning)
        )                                                             # (R,M,H,W) in (0, 2pi*fmax)
        phi = self.phi_projection(conditioning)                       # (R,M,H,W)
        relative = (time_s - source_parameters[:, None, 3]) / float(domain_t_s)  # (R,T)

        scale = 1.0 / math.sqrt(float(self.modes))
        out = time_s.new_zeros((records, count, height, w))
        for start in range(0, count, self.frame_chunk):
            stop = min(start + self.frame_chunk, count)
            rel = relative[:, start:stop]                             # (R,Tc)
            # argument (R,Tc,M,H,W): per-frame time * per-pixel frequency + per-pixel phase
            arg = omega[:, None] * rel[:, :, None, None, None] + phi[:, None]
            modal = amp[:, None] * torch.cos(arg)                     # (R,Tc,M,H,W)
            out[:, start:stop] = modal.sum(dim=2) * scale             # (R,Tc,H,W)
        return self.gate * out                                       # (R,T,H,W)


class _WindowedCharacteristicPropagation(nn.Module):
    """Contract-safe small-window space-time propagation coupler (candidate C1).

    Every prior temporal module (B1 ``_ContinuousTimePropagationBasis``, A3
    ``_ContinuousTemporalLatentBasis``, A5 ``_DispersiveModalField``) corrects frame
    ``i`` from frame ``i``'s OWN scalar time and render -- diagonal in the queried-time
    axis, so no propagation dynamics tie the wavefront position across saved times.
    That per-frame-independent structure is the root cause of the ~0.24 aggregate floor
    and the 0.125 overfit floor (position jitter).

    C1 introduces the first genuine reduce over a *deterministically-constructed*
    neighbor-time window.  For output frame ``i`` at saved index ``k_i`` it builds a
    window of ``D = 2w+1`` neighbor saved indices ``clamp(k_i + delta, 0, N-1)`` --- a
    function of ``k_i`` ALONE, never of which other frames are in the batch.  It then
    advects frame ``i``'s own (already warp-corrected) field along the eikonal
    characteristic ``grad(T)`` to each neighbor time ``t_win`` (no extra U-Net render):

        u(x, t_win) = field_in( x - (t_win - t_i) * grad(T)/||grad(T)||^2 ),

    giving ``D`` space-time snapshots.  A learned bilinear space-time operator over the
    (window x basis) axes -- both query-independent -- combines them into the frame-``i``
    correction:

        S      = spatial_basis(rendered)                      # (R,T,K,H,W)
        coeff  = time_trunk(features(Delta_t))                # (R,T,D,K)
        coupled = einsum('rtdk,rtkhw,rtdhw->rthw', coeff, S, u) / sqrt(K*D)
        out    = field_in + gate * coupled

    WHY this fixes position/phase where B1's 1x1 cannot: B1 only rescales amplitude at
    the location frame ``i`` already placed; C1 physically ADVECTS the field along the
    ray (a continuous shift per Delta_t), and the window operator learns which advected
    neighbor snapshot(s) compose the center frame -- tying the front position across
    saved times.  This also expresses later arrivals / multiples as delayed
    (Delta_t > 0) advected copies, densely and continuously (cf. A4's discrete paths).

    Query invariance holds BY CONSTRUCTION: the window ``k_win`` depends only on ``k_i``;
    ``u`` only on ``field_in[:, i]`` (frame ``i`` itself; the warp is query-invariant)
    and the static ``arrival``; ``coeff`` only on frame ``i``'s ``Delta_t``; ``S`` only
    on frame ``i``'s render.  So a single-frame (count=1) query of frame ``i`` rebuilds
    an identical window and yields a bit-identical output in eval mode.  The
    frame-chunked loop bounds the (R, chunk, D, H, W) intermediate so the 401-frame
    validation panel cannot blow up memory.  ``gate_init == 0`` is an exact warm-start
    no-op (the center tap Delta_t=0 reproduces field_in); launch with a small positive
    gate_init to avoid the A4 cold-start deadlock -- the space/time sub-nets are NOT
    zero-init, so they receive real gradient the moment the gate opens.
    """

    def __init__(
        self,
        width: int,
        *,
        window: int = 2,
        stride: int = 1,
        rank: int = 8,
        max_advect_cells: float = 8.0,
        harmonics: int = 4,
        gate_init: float = 0.0,
        frame_chunk: int = 64,
    ) -> None:
        super().__init__()
        if width <= 0 or rank <= 0:
            raise ValueError("windowed propagation width/rank must be positive")
        if window < 1:
            raise ValueError("windowed propagation window must be >= 1")
        if stride < 1:
            raise ValueError("windowed propagation stride must be >= 1")
        if harmonics < 0:
            raise ValueError("windowed propagation harmonics must be >= 0")
        if max_advect_cells <= 0.0:
            raise ValueError("windowed propagation max_advect_cells must be > 0")
        if gate_init < 0.0:
            raise ValueError("windowed propagation gate_init must be >= 0")
        if frame_chunk <= 0:
            raise ValueError("windowed propagation frame_chunk must be positive")
        self.window = int(window)
        self.stride = int(stride)
        self.rank = int(rank)
        self.harmonics = int(harmonics)
        self.max_advect_cells = float(max_advect_cells)
        self.frame_chunk = int(frame_chunk)
        # per-frame low-rank spatial basis from the rendered features (RF=5).
        self.spatial_basis = nn.Sequential(
            nn.Conv2d(width, self.rank, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.rank, self.rank, kernel_size=3, padding=1),
        )
        # continuous-time trunk over the window offset Delta_t features -> rank coeffs.
        feature_dim = 1 + 2 * self.harmonics + 2  # Delta_t, sin/cos harmonics, Delta_t^2/^3
        self.time_trunk = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.GELU(),
            nn.Linear(width, self.rank),
        )
        # gate_init == 0 -> exact warm-start no-op (center tap reproduces field_in).
        # A small positive value breaks the A4 cold-start deadlock; the sub-nets above
        # are randomly initialized (NOT zero) so they train once the gate opens.
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    @staticmethod
    def _characteristic_direction(arrival: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad_z, grad_x = torch.gradient(arrival, dim=(-2, -1))
        norm_sq = grad_x * grad_x + grad_z * grad_z + 1e-6
        # velocity-along-ray characteristic: dx/dt = grad(T)/||grad(T)||^2
        return grad_x / norm_sq, grad_z / norm_sq

    def _time_features(self, delta_t: torch.Tensor, frequency: torch.Tensor, domain_t_s: float) -> torch.Tensor:
        # delta_t: (R,Tc,D)  frequency: (R,1) source f0 in Hz
        norm = delta_t / float(domain_t_s)
        feats = [norm]
        phase = 2.0 * math.pi * frequency[:, :, None] * delta_t  # (R,Tc,D)
        for h in range(1, self.harmonics + 1):
            feats.append(torch.sin(h * phase))
            feats.append(torch.cos(h * phase))
        feats.append(norm * norm)
        feats.append(norm * norm * norm)
        return torch.stack(feats, dim=-1)  # (R,Tc,D,feature_dim)

    def forward(
        self,
        field_in: torch.Tensor,        # (records, count, H, W) -- already warp-corrected
        rendered: torch.Tensor,        # (records*count, width, H, W)
        arrival: torch.Tensor,         # (records, H, W) static eikonal T(x,z)
        time_s: torch.Tensor,          # (records, count)
        saved_time_indices: torch.Tensor,   # (records, count) long
        saved_time_values: torch.Tensor,    # (N,) stored-time axis in seconds
        source_parameters: torch.Tensor,    # (records, 5) -> f0 at [:,2]
        *,
        domain_t_s: float,
    ) -> torch.Tensor:
        records, count, height, w = field_in.shape
        n_saved = int(saved_time_values.shape[0])
        # spatial basis per queried frame from its own render.
        basis = self.spatial_basis(rendered).reshape(records, count, self.rank, height, w)
        # normalized-coordinate identity grid, layout (x, z) = (W, H).
        device = field_in.device
        dtype = field_in.dtype
        zs = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        base_z, base_x = torch.meshgrid(zs, xs, indexing="ij")           # (H,W)
        d_x, d_z = self._characteristic_direction(arrival)               # (R,H,W)
        frequency = source_parameters[:, 2:3]                            # (R,1)
        # window offsets in saved-index units (function of k_i alone).
        offsets = torch.arange(
            -self.window, self.window + 1, device=device
        ) * self.stride                                                  # (D,)
        n_win = int(offsets.shape[0])
        scale = 1.0 / math.sqrt(float(self.rank * n_win))
        out = field_in.new_zeros((records, count, height, w))
        for start in range(0, count, self.frame_chunk):
            stop = min(start + self.frame_chunk, count)
            tc = stop - start
            k = saved_time_indices[:, start:stop]                        # (R,Tc)
            k_win = torch.clamp(k[:, :, None] + offsets[None, None], 0, n_saved - 1)  # (R,Tc,D)
            # gather neighbor stored times; clamped edges reuse the boundary time.
            # cast to the field dtype (saved_time_values may be float64) so the grid
            # and coeff features match field_in for grid_sample / einsum.
            t_win = saved_time_values[k_win].to(dtype)                   # (R,Tc,D)
            delta_t = t_win - time_s[:, start:stop, None].to(dtype)      # (R,Tc,D)  center tap = 0
            # advect frame i's own field to each neighbor time along grad(T).
            # sample_coord = identity - Delta_t * (grad(T)/||grad(T)||^2), cell->normalized.
            shift_cells = torch.clamp(
                delta_t / float(domain_t_s), -1.0, 1.0
            ) * self.max_advect_cells                                    # (R,Tc,D) bounded
            # broadcast to (R,Tc,D,H,W)
            sx = shift_cells[:, :, :, None, None] * d_x[:, None, None]   # (R,Tc,D,H,W)
            sz = shift_cells[:, :, :, None, None] * d_z[:, None, None]
            sample_x = base_x[None, None, None] - sx * (2.0 / max(w - 1, 1))
            sample_z = base_z[None, None, None] - sz * (2.0 / max(height - 1, 1))
            grid = torch.stack((sample_x, sample_z), dim=-1).reshape(
                records * tc * n_win, height, w, 2
            )
            src = (
                field_in[:, start:stop, None]
                .expand(records, tc, n_win, height, w)
                .reshape(records * tc * n_win, 1, height, w)
            )
            u = F.grid_sample(
                src, grid, mode="bilinear", padding_mode="border", align_corners=True
            ).reshape(records, tc, n_win, height, w)                     # (R,Tc,D,H,W)
            feats = self._time_features(delta_t, frequency, domain_t_s)  # (R,Tc,D,Fdim)
            coeff = self.time_trunk(feats)                               # (R,Tc,D,rank)
            coupled = torch.einsum(
                "rtdk,rtkhw,rtdhw->rthw", coeff, basis[:, start:stop], u
            ) * scale                                                    # (R,Tc,H,W)
            out[:, start:stop] = coupled
        return field_in + self.gate * out


class _MultiScaleSpectralBypass(nn.Module):
    """MscaleFNO-style multi-scale spectral bypass to break the U-Net's rank collapse.

    Diagnosis (VERDICT sec 27): the Helmholtz render's INPUT has effective rank ~66 (rich
    phase features) but the local-conv U-Net collapses its OUTPUT to effective rank ~8 due
    to spectral bias, while the late scattering residual needs rank ~37. The literature
    (MscaleFNO 2024/2026) shows parallel spectral branches at different mode scales recover
    the high-frequency / high-rank components a local U-Net loses.

    This module maps the render INPUT (rank ~66) through several parallel factorized
    spectral convolutions at increasing mode counts (multi-scale), sums them, and returns a
    correction ADDED to the U-Net render output via a ZERO-init gate -- so at construction
    the bypass contributes exactly nothing and the model reproduces A+1 bit-for-bit
    (continue-pretraining contract). Trained, it injects the high-rank spectral structure
    the U-Net dropped, straight from the already-high-rank input.
    """

    def __init__(self, width: int, *, mode_scales: Sequence[int] = (8, 16, 32, 48, 64),
                 gate_init: float = 0.1, per_branch_gate: bool = False) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("spectral bypass width must be positive")
        self.mode_scales = tuple(int(m) for m in mode_scales)
        self.branches = nn.ModuleList(
            AxisFactorizedComplexSpectralConv2d(width, width, modes=int(m)) for m in self.mode_scales
        )
        # 1x1 mixing of the concatenated multi-scale branch outputs back to width.
        # The mix keeps its default (nonzero) initialization. The gate defaults to a SMALL
        # POSITIVE value (0.1), not zero: a zero gate makes the bypass a warm-start no-op but
        # its gradient (dL/d(gate) = mix_out * ...) starts tiny, so under warm-started
        # continued pretraining the spectral branches learn far too slowly to lift the render
        # rank in practice (measured: 96 steps left gate=2.6e-4, render rank 8->9). A small
        # positive gate lets the spectral branches contribute and receive real gradient from
        # step 0; the warm start is then near-A+1 (small perturbation), and held-out
        # monitoring guards against degradation (the gate can shrink back if it hurts).
        # gate_init=0 remains available (flag) as the exact-no-op control.
        # mode_scales spans 8..64: the higher modes (48/64) cover the late scattering coda's
        # wavenumber content (|k| up to ~40) that the local U-Net render collapses out.
        self.mix = nn.Conv2d(width * len(self.branches), width, kernel_size=1)
        self.per_branch_gate = bool(per_branch_gate)
        if self.per_branch_gate:
            # INCREMENTAL / progressive-frequency mode (borrowing neuraloperator's
            # incremental FNO idea: let higher-frequency modes enter gradually rather than
            # all at once). Each mode-scale branch gets its OWN learnable gate, applied to
            # that branch BEFORE the mix, with a FREQUENCY-DECAYING init (low-mode branches
            # start with a larger gate, high-mode branches near zero). The optimizer then
            # raises the high-frequency branches adaptively as training progresses, instead
            # of a single scalar gate that must fight spectral bias to open all modes at
            # once. gate_init sets the largest (lowest-mode) branch; higher modes decay
            # geometrically so the warm start is a low-frequency perturbation of A+1.
            n = len(self.branches)
            inits = [float(gate_init) * (0.5 ** i) for i in range(n)]  # low modes larger
            self.branch_gates = nn.Parameter(torch.tensor(inits, dtype=torch.float32))
            self.gate = None
        else:
            self.gate = nn.Parameter(torch.full((), float(gate_init)))
            self.branch_gates = None

    def forward(self, record_input: torch.Tensor) -> torch.Tensor:
        outs = [branch(record_input) for branch in self.branches]
        if self.per_branch_gate:
            # gate each mode-scale branch independently (progressive-frequency), then mix
            gated = [self.branch_gates[i] * o for i, o in enumerate(outs)]
            return self.mix(torch.cat(gated, dim=1))
        return self.gate * self.mix(torch.cat(outs, dim=1))


class _DirectFrequencySpectralExperts2d(nn.Module):
    """Medium-conditioned full 2-D transfer kernels for complex Born frequencies.

    A single translation-invariant Green multiplier cannot serve the layered and
    Marmousi records simultaneously: their measured least-squares transfer functions
    conflict even though each record is well represented by a 48-mode 2-D kernel.
    This branch therefore mixes a small bank of per-frequency complex kernels using
    a record summary from the already conditioned Born features.

    The kernels are zero-initialised, so enabling the branch is an exact warm-start
    no-op.  Unlike a zero scalar gate, the kernels themselves receive a useful
    gradient on the first update.  The Born source is RMS-normalised per record only
    to keep this new optimizer group numerically identifiable; its scale is included
    in the expert-gating summary.
    """

    def __init__(
        self,
        conditioning_width: int,
        *,
        num_frequencies: int,
        modes: int,
        experts: int,
    ) -> None:
        super().__init__()
        width = int(conditioning_width)
        frequencies = int(num_frequencies)
        mode_count = int(modes)
        expert_count = int(experts)
        if width <= 0 or frequencies <= 0 or mode_count <= 0 or expert_count <= 0:
            raise ValueError("direct spectral expert dimensions must be positive")
        self.conditioning_width = width
        self.num_frequencies = frequencies
        self.modes = mode_count
        self.experts = expert_count
        # Four signed (kz,kx) quadrants.  The last pair stores real/imaginary parts.
        self.kernel = nn.Parameter(
            torch.zeros(
                expert_count,
                frequencies,
                4,
                mode_count,
                mode_count,
                2,
                dtype=torch.float32,
            )
        )
        self.expert_gate = nn.Linear(width + 1, expert_count)
        # A 4x4 signed/absolute contrast thumbnail distinguishes layered interfaces
        # from Marmousi structure.  Zero initialisation preserves both legacy
        # checkpoints and an already-trained spectral-expert checkpoint exactly;
        # once the expert kernels differ, this router receives a first-step gradient.
        self.contrast_expert_gate = nn.Linear(2 * 4 * 4, expert_count, bias=False)
        nn.init.zeros_(self.contrast_expert_gate.weight)

    def forward(
        self,
        born: torch.Tensor,          # complex [record,frequency,z,x]
        conditioning: torch.Tensor,  # real [record,width,z,x]
        contrast: torch.Tensor,      # real [record,z,x]
    ) -> torch.Tensor:
        if not torch.is_complex(born) or born.ndim != 4:
            raise ValueError("direct spectral Born source must be complex [record,f,z,x]")
        records, frequencies, height, width = born.shape
        if frequencies != self.num_frequencies:
            raise ValueError("direct spectral Born frequency count is invalid")
        if conditioning.shape != (
            records,
            self.conditioning_width,
            height,
            width,
        ):
            raise ValueError("direct spectral conditioning shape is invalid")
        if contrast.shape != (records, height, width):
            raise ValueError("direct spectral contrast shape is invalid")

        with torch.autocast(device_type=born.device.type, enabled=False):
            source = born.to(torch.complex64)
            source_rms = source.abs().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
            source_scale = source_rms.clamp_min(1.0e-6)
            normalized = source / source_scale
            summary = conditioning.float().mean(dim=(-2, -1))
            log_scale = torch.log10(source_scale[:, 0, 0, 0]).clamp(-6.0, 2.0)
            contrast32 = contrast.float()
            contrast_peak = contrast32.abs().flatten(1).amax(dim=1).clamp_min(1.0e-6)
            relative_contrast = contrast32 / contrast_peak[:, None, None]
            contrast_summary = torch.cat(
                (
                    F.adaptive_avg_pool2d(relative_contrast[:, None], (4, 4)).flatten(1),
                    F.adaptive_avg_pool2d(relative_contrast.abs()[:, None], (4, 4)).flatten(1),
                ),
                dim=1,
            )
            logits = self.expert_gate(
                torch.cat((summary, log_scale[:, None]), dim=1)
            ) + self.contrast_expert_gate(contrast_summary)
            mixture = torch.softmax(
                logits, dim=1,
            )
            kernels = torch.view_as_complex(self.kernel.contiguous())
            mixed = torch.einsum(
                "be,efqzx->bfqzx", mixture.to(kernels.dtype), kernels
            )

            spectrum = torch.fft.fft2(normalized, dim=(-2, -1), norm="ortho")
            output_spectrum = torch.zeros_like(spectrum)
            modes_z = min(self.modes, height // 2)
            modes_x = min(self.modes, width // 2)
            if modes_z <= 0 or modes_x <= 0:
                raise ValueError("direct spectral expert grid is too small")
            output_spectrum[:, :, :modes_z, :modes_x] = (
                spectrum[:, :, :modes_z, :modes_x]
                * mixed[:, :, 0, :modes_z, :modes_x]
            )
            output_spectrum[:, :, :modes_z, -modes_x:] = (
                spectrum[:, :, :modes_z, -modes_x:]
                * mixed[:, :, 1, :modes_z, :modes_x]
            )
            output_spectrum[:, :, -modes_z:, :modes_x] = (
                spectrum[:, :, -modes_z:, :modes_x]
                * mixed[:, :, 2, :modes_z, :modes_x]
            )
            output_spectrum[:, :, -modes_z:, -modes_x:] = (
                spectrum[:, :, -modes_z:, -modes_x:]
                * mixed[:, :, 3, :modes_z, :modes_x]
            )
            propagated = torch.fft.ifft2(
                output_spectrum, dim=(-2, -1), norm="ortho"
            )

            # Match synthesize_direct_frequency_output's cosine/sine convention.
            cosine = 2.0 * propagated.real
            sine = -2.0 * propagated.imag
            cosine[:, 0] = propagated[:, 0].real
            sine[:, 0] = 0.0
            return torch.cat((cosine, sine), dim=1)


class _BackgroundBornConditioner(nn.Module):
    """Condition the learned scattering field on ``P_bg`` and medium contrast.

    The A+1 training path previously used ``P_bg`` only as an additive baseline:
    targets were changed from ``p`` to ``p - P_bg``, but the neural corrector never
    saw the incident/background field.  That omits the defining input of the
    scattering problem.  In the frequency domain a first-order source has the form

        q_j(x) = omega_j^2 * (m(x) - m_bg(x)) * P_bg,j(x),

    where ``m = v^-2``.  This module constructs a dimensionless version of that
    source from the complete stored-time background field, WKB-demodulates it with
    the same eikonal phase used by the Helmholtz synthesis head, and projects its
    real/imaginary frequency channels into the render width.

    The last projection is zero-initialised.  Consequently a checkpoint that did
    not contain this module is reproduced exactly at warm start, while the final
    projection receives a nonzero gradient immediately.  The input is the fixed
    complete stored-time ``P_bg`` axis, independent of the requested frame subset,
    so single-frame and multi-frame queries use identical conditioning.
    """

    def __init__(
        self,
        width: int,
        *,
        num_frequencies: int,
        background_sigma_cells: float = 2.0,
        global_propagation: bool = False,
        propagation_modes: int = 48,
        direct_frequency_output: bool = False,
        direct_frequency_count: int = 0,
        direct_spectral_experts: int = 0,
    ) -> None:
        super().__init__()
        if width <= 0 or num_frequencies <= 0:
            raise ValueError("background Born conditioner dimensions must be positive")
        if background_sigma_cells <= 0.0:
            raise ValueError("background smoothing sigma must be positive")
        if propagation_modes <= 0:
            raise ValueError("background propagation modes must be positive")
        self.direct_frequency_output = bool(direct_frequency_output)
        requested_direct_frequencies = int(direct_frequency_count)
        if self.direct_frequency_output and requested_direct_frequencies > 0:
            if requested_direct_frequencies > int(num_frequencies):
                raise ValueError(
                    "direct Born frequency count cannot exceed parent synthesis frequencies"
                )
            self.num_frequencies = requested_direct_frequencies
        else:
            self.num_frequencies = int(num_frequencies)
        self.background_sigma_cells = float(background_sigma_cells)
        self.global_propagation = bool(global_propagation)
        self.direct_spectral_expert_count = int(direct_spectral_experts)
        if self.direct_spectral_expert_count < 0:
            raise ValueError("direct spectral expert count must be nonnegative")
        if self.direct_spectral_expert_count > 0 and not self.direct_frequency_output:
            raise ValueError("direct spectral experts require direct frequency output")
        if self.direct_frequency_output and not self.global_propagation:
            raise ValueError("direct Born frequency output requires global propagation")

        radius = max(1, int(math.ceil(3.0 * self.background_sigma_cells)))
        coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(
            -0.5 * (coordinate / self.background_sigma_cells).square()
        )
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer(
            "background_smoothing_kernel",
            kernel_2d[None, None],
            persistent=True,
        )

        # 2*nf channels are Re/Im of the WKB-demodulated Born source; the final
        # channel is the dimensionless slowness contrast itself.
        self.input_projection = nn.Sequential(
            nn.Conv2d(2 * self.num_frequencies + 1, width, kernel_size=1),
            _group_norm(width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )
        if self.global_propagation:
            # A compact trainable scattering propagator.  The spectral paths provide
            # global support while the local paths retain interface-scale structure.
            # Its output is injected AFTER the frozen parent U-Net, so it cannot be
            # normalized away by that renderer (the failure isolated at update 96).
            self.propagation_spectral = nn.ModuleList(
                [
                    AxisFactorizedComplexSpectralConv2d(
                        width, width, modes=int(propagation_modes)
                    )
                    for _ in range(2)
                ]
            )
            self.propagation_local = nn.ModuleList(
                [nn.Conv2d(width, width, kernel_size=3, padding=1) for _ in range(2)]
            )
            self.propagation_norm = nn.ModuleList(
                [_group_norm(width) for _ in range(2)]
            )
            # The legacy "global" path sums independent x-only and z-only spectral
            # contractions.  That can spread a compact Born source along rows and
            # columns, but it has no direct joint (kz,kx) response -- exactly the
            # structure needed for oblique reflection propagation.  Reuse each
            # factorized layer's coupled x-then-z operator behind a zero-initialised
            # gate.  Existing checkpoints therefore remain function-identical at
            # load, while reflection recovery can learn a genuine separable 2-D
            # Green response without adding a large new spectral tensor.
            self.propagation_coupled_gates = nn.Parameter(
                torch.zeros(len(self.propagation_spectral), dtype=torch.float32)
            )
            output_channels = (
                2 * self.num_frequencies
                if self.direct_frequency_output
                else width
            )
            self.output_projection = nn.Conv2d(width, output_channels, kernel_size=1)
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
            self.direct_spectral_propagator = (
                _DirectFrequencySpectralExperts2d(
                    width,
                    num_frequencies=self.num_frequencies,
                    modes=int(propagation_modes),
                    experts=self.direct_spectral_expert_count,
                )
                if self.direct_spectral_expert_count > 0
                else None
            )
        else:
            # Legacy input-injection probe: exact parent reproduction at construction.
            nn.init.zeros_(self.input_projection[-1].weight)
            nn.init.zeros_(self.input_projection[-1].bias)
            self.propagation_spectral = None
            self.propagation_local = None
            self.propagation_norm = None
            self.register_parameter("propagation_coupled_gates", None)
            self.output_projection = None
            self.direct_spectral_propagator = None

    def slowness_contrast(self, velocity_mps: torch.Tensor) -> torch.Tensor:
        """Return ``v_bg^2 * (v^-2 - v_bg^-2)`` on the saved spatial grid."""

        velocity = torch.as_tensor(velocity_mps)
        if velocity.ndim == 4 and velocity.shape[1] == 1:
            velocity = velocity[:, 0]
        if velocity.ndim != 3:
            raise ValueError(
                "background conditioner velocity must be [record,H,W] or [record,1,H,W]"
            )
        if not bool(torch.all(velocity > 0.0)):
            raise ValueError("background conditioner velocity must be positive")
        radius = self.background_smoothing_kernel.shape[-1] // 2
        padded = F.pad(
            velocity[:, None],
            (radius, radius, radius, radius),
            mode="replicate",
        )
        kernel = self.background_smoothing_kernel.to(
            device=velocity.device, dtype=velocity.dtype
        )
        background = F.conv2d(padded, kernel)[:, 0].clamp_min(1.0)
        contrast = background.square() / velocity.square() - 1.0
        # A constant field remains constant analytically; remove float32 convolution
        # roundoff so the uniform-family zero-scattering invariant is exact in code.
        return torch.where(contrast.abs() < 1.0e-5, torch.zeros_like(contrast), contrast)

    def forward(
        self,
        background_normalized: torch.Tensor,  # (records,N,H,W), complete stored axis
        velocity_mps: torch.Tensor,           # (medium,H,W)
        record_to_medium: torch.Tensor,       # (records,)
        arrival: torch.Tensor,                # (records,H,W)
        omega: torch.Tensor,                  # (nf,) rad/s
    ) -> torch.Tensor:
        background = torch.as_tensor(background_normalized)
        if background.ndim != 4:
            raise ValueError("background conditioning field must be [record,time,H,W]")
        records, stored_count, height, width = background.shape
        mapping = torch.as_tensor(
            record_to_medium, device=background.device, dtype=torch.long
        )
        if mapping.shape != (records,):
            raise ValueError("background conditioning record mapping is invalid")
        if arrival.shape != (records, height, width):
            raise ValueError("background conditioning arrival field is invalid")
        frequencies = torch.as_tensor(
            omega, device=background.device, dtype=background.dtype
        )
        if frequencies.ndim != 1 or frequencies.numel() < self.num_frequencies:
            raise ValueError("background conditioning frequency bank is invalid")
        frequencies = frequencies[: self.num_frequencies]
        if stored_count // 2 + 1 < self.num_frequencies:
            raise ValueError("background time axis is too short for requested frequencies")

        spectrum = torch.fft.rfft(background.float(), dim=1, norm="forward")
        spectrum = spectrum[:, : self.num_frequencies]
        # Remove the rapidly oscillating eikonal phase so the learned projection sees
        # smooth complex envelopes, matching the WKB synthesis parametrization.
        phase = frequencies[None, :, None, None].float() * arrival[:, None].float()
        demodulated = (
            spectrum
            if self.direct_frequency_output
            else spectrum * torch.complex(torch.cos(phase), torch.sin(phase))
        )

        velocity = torch.as_tensor(velocity_mps, device=background.device)[mapping]
        contrast = self.slowness_contrast(velocity.float())
        omega_scale = frequencies.float() / frequencies.float().abs().max().clamp_min(1.0)
        born = demodulated * contrast[:, None] * omega_scale[None, :, None, None].square()
        features = torch.cat(
            (born.real, born.imag, contrast[:, None]), dim=1
        ).to(background.dtype)
        # Keep the physical invariant exact: a uniform medium has delta-m == 0 and
        # therefore no learned scattering correction.  Normalising by each record's
        # peak contrast avoids squaring a tiny physical scale for heterogeneous media,
        # while the multiplicative support prevents projection biases from leaking a
        # spurious field into the uniform family.
        raw_peak = contrast.abs().flatten(1).amax(dim=1)
        contrast_peak = raw_peak.clamp_min(1.0e-6)
        support = contrast.abs() / contrast_peak[:, None, None]
        source = self.input_projection(features) * support[:, None].to(background.dtype)
        if not self.global_propagation:
            return source
        assert self.propagation_spectral is not None
        assert self.propagation_local is not None
        assert self.propagation_norm is not None
        assert self.output_projection is not None
        propagated = source
        for layer_index, (spectral, local, norm) in enumerate(zip(
            self.propagation_spectral,
            self.propagation_local,
            self.propagation_norm,
            strict=True,
        )):
            coupled = torch.tanh(self.propagation_coupled_gates[layer_index]) * (
                spectral.coupled(propagated)
            )
            propagated = F.gelu(
                norm(
                    propagated
                    + spectral(propagated)
                    + coupled
                    + local(propagated)
                )
            )
        # The scalar record gate preserves the exact no-scattering invariant for a
        # constant medium without re-localising the globally propagated field.
        record_gate = (raw_peak >= 1.0e-5).to(background.dtype)[:, None, None, None]
        coefficients = self.output_projection(propagated)
        if self.direct_spectral_propagator is not None:
            coefficients = coefficients + self.direct_spectral_propagator(
                born, source, contrast
            )
        return coefficients * record_gate

    def synthesize_direct_frequency_output(
        self,
        coefficients: torch.Tensor,
        time_s: torch.Tensor,
        saved_time_values: torch.Tensor,
        *,
        frame_chunk: int = 64,
    ) -> torch.Tensor:
        """Render raw scattered-frequency coefficients without a direct-ray phase.

        Reflections carry additional path delays, so forcing their coefficients through
        the parent's single eikonal phase anchor makes the envelopes oscillatory and led
        to the observed near-zero solution.  This branch keeps one complex coefficient
        field per frequency and synthesizes it against absolute stored time directly.
        """

        if not self.direct_frequency_output:
            raise ValueError("conditioner is not configured for direct frequency output")
        values = torch.as_tensor(saved_time_values, device=coefficients.device)
        if values.ndim != 1 or values.numel() < 2:
            raise ValueError("direct frequency synthesis needs the complete stored axis")
        records, channels, height, width = coefficients.shape
        if channels != 2 * self.num_frequencies:
            raise ValueError("direct frequency coefficient channel count is invalid")
        times = torch.as_tensor(time_s, device=coefficients.device)
        if times.ndim != 2 or times.shape[0] != records:
            raise ValueError("direct frequency synthesis times must be [record,time]")
        dt = (values[-1] - values[0]) / float(values.numel() - 1)
        omega = (
            2.0
            * math.pi
            * torch.arange(
                self.num_frequencies,
                device=coefficients.device,
                dtype=coefficients.dtype,
            )
            / (float(values.numel()) * dt.to(coefficients.dtype))
        )
        a = coefficients[:, : self.num_frequencies].flatten(2)
        b = coefficients[:, self.num_frequencies :].flatten(2)
        out = coefficients.new_zeros((records, times.shape[1], height, width))
        for start in range(0, times.shape[1], int(frame_chunk)):
            stop = min(start + int(frame_chunk), times.shape[1])
            phase = times[:, start:stop, None].to(coefficients.dtype) * omega[None, None]
            frame = torch.bmm(torch.cos(phase), a) + torch.bmm(torch.sin(phase), b)
            out[:, start:stop] = frame.reshape(records, stop - start, height, width)
        return out


class _HelmholtzSynthesisField(nn.Module):
    """Temporal-frequency (Helmholtz) coarse-field parametrization (candidate H1).

    Every prior coarse-field form renders frame ``i`` from frame ``i``'s OWN scalar
    time (per-frame FiLM + saved-index embedding, or a gated per-frame residual) --
    diagonal in the queried-time axis.  That per-frame-independent placement of a
    moving wavefront is the measured root cause of the ~0.24 aggregate / 0.125 overfit
    floor: the model re-guesses the front position at every saved time, so positions
    jitter across time.  See ``results/capacity_physical_limit_study.md`` (position/
    phase, NOT capacity or Nyquist, is the wall) and the four falsified propagation
    rungs (warp/A3/A4/C1, all first-arrival grad(T) diagonal).

    H1 changes the *time representation* itself.  A temporal Fourier transform decouples
    the sourced wave equation into per-frequency static Helmholtz fields
    ``P(x,z,omega_j)`` (no time axis).  The model predicts ``nf`` such complex spatial
    fields once per record from the *time-independent* conditioning, and ANY saved-time
    frame is reconstructed by the FIXED, non-learned inverse transform

        p(x,z,t) = sum_j [ a_j(x,z) cos(omega_j t) + b_j(x,z) sin(omega_j t) ],

    with ``omega_j = 2*pi*j/(N*dt)`` the rfft grid matching the stored-time axis.

    Why this attacks the position/phase floor the diagonal forms cannot:
      * The wavefront arrival time tau(x) IS the *phase* of P_j (P_j ~ e^{-i omega_j tau}).
        Cross-time coherence of the front is BUILT INTO the exact inverse transform, so
        the model no longer re-places the front per frame -> the jitter is removed by
        construction, not learned away.
      * Regressing static complex fields (no time axis) is structurally simpler than
        placing a moving wavefront across 401 frames, and the phase is anchored to the
        (static) eikonal travel-time field the project already computes.

    Zero-training oracle bounds (``results/temporal_frequency_helmholtz_study.md``,
    held-out records): nf=64 lowest bins, each spatially truncated to |k|<=40, rebuild
    all 401 frames to relL2 uniform 0.045 / layered 0.044 / marmousi 0.047 -- the first
    constructive evidence that <5% is reachable inside the query-invariance contract.

    Query invariance holds BY CONSTRUCTION: ``a_j, b_j`` do not depend on ``t``; each
    output frame uses only its own scalar ``t`` through the analytic basis, so a
    single-frame (count=1) query is bit-identical to that frame inside any batch.  No
    autoregression, no cross-query coupling.  The frame-chunked synthesis bounds the
    ``(R, chunk, H, W)`` intermediate so the 401-frame validation panel cannot blow up.
    """

    def __init__(
        self,
        width: int,
        *,
        num_frequencies: int = 64,
        frame_chunk: int = 64,
        wkb_phase: bool = True,
        rank: int = 0,
        late_rank: int = 0,
        late_frequencies: int = 0,
        frequency_softmax: bool = False,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("helmholtz synthesis width must be positive")
        if num_frequencies < 1:
            raise ValueError("helmholtz synthesis num_frequencies must be >= 1")
        if frame_chunk <= 0:
            raise ValueError("helmholtz synthesis frame_chunk must be positive")
        if rank < 0:
            raise ValueError("helmholtz synthesis rank must be >= 0")
        self.num_frequencies = int(num_frequencies)
        # Optional record-conditioned frequency-attention gate (single structural
        # factor, zero-init => exact identity at construction so any parent
        # checkpoint continues bit-for-bit).  The spectrum-relative-L2 audit shows
        # the current head fits the lowest bins well but leaves the middle/high
        # bands at relL2 ~0.3-0.6.  Pooling the rendered record features and mapping
        # them to nf logits lets different records re-weight their frequency bins;
        # a shared (1,nf,1) parameter would only be a redundant global rescaling of
        # the already-independent output channels.
        self.frequency_softmax = bool(frequency_softmax)
        if self.frequency_softmax:
            self.frequency_gate = nn.Linear(width, self.num_frequencies)
            nn.init.zeros_(self.frequency_gate.weight)
            nn.init.zeros_(self.frequency_gate.bias)
        self.frame_chunk = int(frame_chunk)
        # WKB / geometric-optics ansatz.  A NAIVE Helmholtz head must output the FULLY
        # OSCILLATORY complex field P_j (standing waves at wavelength c/f_j, |k| up to
        # ~40) -- which a local-conv U-Net cannot render (getting the oscillation right
        # means getting the travel-time phase right to sub-wavelength precision, i.e. the
        # SAME position/phase problem).  The G2 probe confirmed the naive head caps at
        # agg~1.1 (worse with more frequencies = temporal underdetermination + oscillatory
        # target).  WKB factors the physics out:  P_j(x) = A_j(x) * exp(-i omega_j tau(x)),
        # with A_j a SMOOTH (low-|k|) amplitude the U-Net CAN produce and the fast
        # oscillation supplied ANALYTICALLY by the eikonal travel time tau.  In time domain
        # the synthesis becomes retarded-time:  p = sum_j a_j cos(w_j (t - tau)) +
        # b_j sin(w_j (t - tau)), so at the wavefront (t ~= tau) all frequencies are in
        # phase -> a sharp pulse whose POSITION is anchored by tau, not re-guessed.
        self.wkb_phase = bool(wkb_phase)
        # Static travel-time phase anchor: one input channel (tau normalized) projected
        # into the conditioning width so the amplitude head can also see tau directly.
        self.phase_anchor = nn.Conv2d(1, width, kernel_size=1)
        # LOW-RANK cross-frequency factorization (rank > 0).  The G2 probe showed a head
        # with 2*nf INDEPENDENT coefficient fields per pixel is temporally UNDERDETERMINED
        # (16 training frames < 2*nf unknowns; more frequencies made it worse).  The
        # amplitude-learnability probe measured the K frequency amplitude fields to have
        # cross-frequency rank <= 6 (marmousi rank 1) -- they share a tiny common basis.
        # So parametrize a_j(x) = sum_r cos_mix[j,r] * B_r(x), b_j(x) = sum_r
        # sin_mix[j,r] * B_r(x): R shared SMOOTH basis fields B_r(x) from the render + a
        # small learned per-frequency mixing.  Per-pixel DOF drops from 2*nf to R,
        # dissolving the underdetermination while keeping the exact analytic time basis.
        self.rank = int(rank)
        if self.rank > 0:
            # R shared SMOOTH basis fields B_r(x) from the render (low-|k|).
            self.basis_head = nn.Conv2d(width, self.rank, kernel_size=1)
            # Per-frequency complex mixing M[nf, R].  A GLOBAL base mixing (shared across
            # records) PLUS a per-RECORD conditioned mixing predicted from the render's
            # global-pooled features.  The design doc specifies M conditioned on source
            # position; the first low-rank G2 with a purely global (record-shared) mixing
            # collapsed to the trivial (near-zero) solution at agg~1.0 because three very
            # different records were forced to share one frequency->basis mixing.  The
            # conditioned term restores per-record freedom while keeping DOF tiny.
            self.cos_mix = nn.Parameter(torch.randn(self.num_frequencies, self.rank) * 0.1)
            self.sin_mix = nn.Parameter(torch.randn(self.num_frequencies, self.rank) * 0.1)
            self.mix_condition = nn.Linear(width, 2 * self.num_frequencies * self.rank)
            nn.init.zeros_(self.mix_condition.weight)
            nn.init.zeros_(self.mix_condition.bias)
        else:
            # Coefficient head: width -> 2*nf channels.  Under WKB these are the SMOOTH
            # amplitude envelopes (a_j = Re A_j, b_j = -Im A_j); without WKB they are the
            # raw (oscillatory) cos/sin coefficients.
            self.head = nn.Conv2d(width, 2 * self.num_frequencies, kernel_size=1)

        # LATE-MULTIPLE dedicated higher-rank basis (late_rank > 0, requires rank > 0).
        # Diagnosis (VERDICT sec 22): the truth-minus-A+1 residual's spatial rank grows
        # monotonically over time (early rank@90%~7 / mid~19 / LATE~32, @99%~51) with 94%
        # of the residual energy in the late window; the shared rank-8 fits the early
        # regime but truncates the late reverberation coda.  Late multiples are dominated
        # by the LOWEST frequency bins (peak bin ~19-23), so we add R_late extra smooth
        # basis fields whose per-frequency mixing is confined to the lowest
        # ``late_frequencies`` bins and ADDED to (a, b) there.  The mixing is ZERO-init so
        # at construction the late head contributes exactly nothing -> the model bit-for-bit
        # reproduces the current A+1 (continue-pretraining contract), and the extra rank is
        # learned only where (low freq / late time) it is actually needed.
        self.late_rank = int(late_rank)
        self.late_frequencies = int(late_frequencies) if late_frequencies > 0 else min(24, self.num_frequencies)
        if self.late_rank > 0:
            if self.rank <= 0:
                raise ValueError("helmholtz late_rank requires the low-rank path (rank > 0)")
            if self.late_frequencies > self.num_frequencies:
                raise ValueError("late_frequencies cannot exceed num_frequencies")
            self.late_basis_head = nn.Conv2d(width, self.late_rank, kernel_size=1)
            # Zero-init per-frequency mixing over the lowest late_frequencies bins.
            self.late_cos_mix = nn.Parameter(torch.zeros(self.late_frequencies, self.late_rank))
            self.late_sin_mix = nn.Parameter(torch.zeros(self.late_frequencies, self.late_rank))
            # Per-record conditioned mixing (zero-init), mirroring self.mix_condition.
            self.late_mix_condition = nn.Linear(width, 2 * self.late_frequencies * self.late_rank)
            nn.init.zeros_(self.late_mix_condition.weight)
            nn.init.zeros_(self.late_mix_condition.bias)

    def frequency_bank(self, saved_time_values: torch.Tensor) -> torch.Tensor:
        """rfft angular-frequency grid for the lowest ``nf`` bins (rad/s)."""

        n_saved = int(saved_time_values.shape[0])
        if n_saved < 2:
            raise ValueError("helmholtz synthesis needs at least two stored times")
        dt = (saved_time_values[-1] - saved_time_values[0]) / float(n_saved - 1)
        df = 1.0 / (float(n_saved) * dt)  # matches numpy.fft.rfftfreq(N, d=dt)
        j = torch.arange(self.num_frequencies, device=saved_time_values.device, dtype=saved_time_values.dtype)
        return (2.0 * math.pi) * (j * df)  # (nf,)

    def frequency_gate_weights(self, rendered: torch.Tensor) -> torch.Tensor:
        """Return record-conditioned identity-normalized frequency weights."""

        if not self.frequency_softmax:
            raise RuntimeError("frequency gate weights require frequency_softmax")
        if rendered.ndim != 4:
            raise ValueError("Helmholtz rendered features must be [record,channel,z,x]")
        scores = self.frequency_gate(rendered.mean(dim=(-2, -1)))
        return torch.softmax(scores, dim=1) * float(self.num_frequencies)

    def apply_frequency_gate_to_coefficients(
        self,
        coefficients: torch.Tensor,
        rendered: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the synthesis gate to raw [cos,sin] coefficient fields."""

        if coefficients.ndim != 4 or coefficients.shape[1] != 2 * self.num_frequencies:
            raise ValueError("Helmholtz coefficients must be [record,2*frequency,z,x]")
        if (
            coefficients.shape[0] != rendered.shape[0]
            or coefficients.shape[-2:] != rendered.shape[-2:]
        ):
            raise ValueError("Helmholtz coefficients and rendered features do not align")
        gate = self.frequency_gate_weights(rendered).to(coefficients.dtype)
        gate = gate[:, :, None, None]
        cosine, sine = coefficients.split(self.num_frequencies, dim=1)
        return torch.cat((cosine * gate, sine * gate), dim=1)

    def forward(
        self,
        rendered: torch.Tensor,        # (records, width, H, W) -- ONE render per record
        arrival: torch.Tensor,         # (records, H, W) static eikonal travel time (s)
        time_s: torch.Tensor,          # (records, count) absolute saved times (s)
        saved_time_values: torch.Tensor,  # (N,) stored-time axis (s)
        *,
        domain_t_s: float,
        late_record_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        records, _, height, w = rendered.shape
        count = int(time_s.shape[1])
        nf = self.num_frequencies
        if self.rank > 0:
            # low-rank: R shared smooth basis fields, mixed per frequency.
            basis = self.basis_head(rendered).reshape(records, self.rank, height * w)  # (R,rank,HW)
            # per-record conditioned mixing (zero-init -> starts at the shared base mixing).
            pooled = rendered.mean(dim=(-2, -1))                                        # (R,width)
            delta = self.mix_condition(pooled).reshape(
                records, 2, self.num_frequencies, self.rank
            )                                                                           # (R,2,nf,rank)
            cos_mix = self.cos_mix[None] + delta[:, 0]                                  # (R,nf,rank)
            sin_mix = self.sin_mix[None] + delta[:, 1]
            a = torch.einsum("bjr,brs->bjs", cos_mix.to(rendered.dtype), basis)         # (R,nf,HW)
            b = torch.einsum("bjr,brs->bjs", sin_mix.to(rendered.dtype), basis)         # (R,nf,HW)
            if self.late_rank > 0:
                # Dedicated higher-rank late-multiple basis, ADDED to the lowest
                # late_frequencies bins.  Zero-init mixing -> no-op at construction.
                lf = self.late_frequencies
                late_basis = self.late_basis_head(rendered).reshape(
                    records, self.late_rank, height * w
                )                                                                       # (R,Rl,HW)
                late_delta = self.late_mix_condition(pooled).reshape(
                    records, 2, lf, self.late_rank
                )                                                                       # (R,2,lf,Rl)
                late_cos = self.late_cos_mix[None] + late_delta[:, 0]                    # (R,lf,Rl)
                late_sin = self.late_sin_mix[None] + late_delta[:, 1]
                a_late = torch.einsum("bjr,brs->bjs", late_cos.to(rendered.dtype), late_basis)
                b_late = torch.einsum("bjr,brs->bjs", late_sin.to(rendered.dtype), late_basis)
                if late_record_gate is not None:
                    record_gate = torch.as_tensor(
                        late_record_gate,
                        device=rendered.device,
                        dtype=rendered.dtype,
                    )
                    if record_gate.shape != (records,):
                        raise ValueError("Helmholtz late-record gate must have shape [records]")
                    a_late = a_late * record_gate[:, None, None]
                    b_late = b_late * record_gate[:, None, None]
                a = a.clone()
                b = b.clone()
                a[:, :lf] = a[:, :lf] + a_late
                b[:, :lf] = b[:, :lf] + b_late
        else:
            coeffs = self.head(rendered)  # (R, 2*nf, H, W)
            if self.frequency_softmax:
                coeffs = self.apply_frequency_gate_to_coefficients(coeffs, rendered)
            a = coeffs[:, :nf].reshape(records, nf, height * w)          # (R, nf, HW)
            b = coeffs[:, nf:].reshape(records, nf, height * w)          # (R, nf, HW)
        if self.frequency_softmax and self.rank > 0:
            # softmax(scores)=1/nf at zero-init -> gate = nf * 1/nf = 1 (identity).
            gate = self.frequency_gate_weights(rendered).to(a.dtype)[:, :, None]
            a = a * gate
            b = b * gate
        omega = self.frequency_bank(saved_time_values).to(rendered.dtype)  # (nf,)
        if self.wkb_phase:
            # Retarded-time synthesis:  p(t) = sum_j a_j cos(w_j (t - tau)) +
            # b_j sin(w_j (t - tau)).  The eikonal travel time tau supplies the fast
            # oscillation analytically, so a_j/b_j only need to be SMOOTH envelopes.
            tau = arrival.reshape(records, 1, height * w)           # (R,1,HW)
        out = rendered.new_zeros((records, count, height, w))
        for start in range(0, count, self.frame_chunk):
            stop = min(start + self.frame_chunk, count)
            t = time_s[:, start:stop].to(rendered.dtype)            # (R, Tc)
            if self.wkb_phase:
                # per-(frame,pixel) retarded phase w_j (t - tau); a_j/b_j are constant in
                # t, so sum over j with an elementwise cos/sin then reduce over frequency.
                # arg: (R, Tc, nf, HW)
                t_minus_tau = t[:, :, None, None] - tau[:, None]     # (R,Tc,1,HW)
                arg = omega[None, None, :, None] * t_minus_tau       # (R,Tc,nf,HW)
                frame = (
                    a[:, None] * torch.cos(arg) + b[:, None] * torch.sin(arg)
                ).sum(dim=2)                                         # (R,Tc,HW)
            else:
                arg = omega[None, None, :] * t[:, :, None]          # (R, Tc, nf)
                cos_jt = torch.cos(arg)
                sin_jt = torch.sin(arg)
                # p = cos_jt @ a + sin_jt @ b  (batched matmul over records)
                frame = torch.bmm(cos_jt, a) + torch.bmm(sin_jt, b)  # (R, Tc, HW)
            out[:, start:stop] = frame.reshape(records, stop - start, height, w)
        return out


class LocalPropagationFieldGenerator(nn.Module):
    """Produce the coarse normalized-pressure field for requested saved times.

    Output shape ``[records, count, height, width]`` matches the tensor the dense
    decoder already expects, so the factorized spectral corrector stays a residual on
    top and every downstream contract (free-surface factor, stored-index query) is
    preserved.
    """

    def __init__(
        self,
        *,
        width: int,
        pyramid_levels: int,
        saved_time_count: int,
        domain_t_s: float,
        domain_diagonal_m: float,
        domain_x_m: float | None = None,
        domain_z_m: float | None = None,
        gabor_scales_s: Sequence[float] = (0.01, 0.025, 0.05, 0.1),
        channel_multipliers: Sequence[int] = (1, 1, 2, 2),
        causal_width_s: float = 0.005,
        residual: bool = False,
        activation_checkpointing: bool = True,
        extended_late_features: bool = False,
        temporal_operator_rank: int = 0,
        temporal_operator_spatial_kernel: int = 1,
        warp: bool = False,
        warp_max_shift_cells: float = 8.0,
        warp_shift_dilation: int = 1,
        green_kernel: bool = False,
        green_kernel_size: int = 5,
        green_dilations: Sequence[int] = (1, 2, 4),
        temporal_latent_basis: bool = False,
        temporal_latent_rank: int = 8,
        temporal_latent_harmonics: int = 4,
        dispersive_modal: bool = False,
        dispersive_modal_modes: int = 16,
        dispersive_modal_max_frequency: float = 8.0,
        multi_arrival: bool = False,
        multi_arrival_paths: int = 3,
        multi_arrival_max_shift_cells: float = 8.0,
        multi_arrival_max_delay_frac: float = 0.5,
        windowed_propagation: bool = False,
        windowed_propagation_window: int = 2,
        windowed_propagation_stride: int = 1,
        windowed_propagation_rank: int = 8,
        windowed_propagation_max_advect_cells: float = 8.0,
        helmholtz_synthesis: bool = False,
        helmholtz_synthesis_frequencies: int = 64,
        helmholtz_synthesis_wkb_phase: bool = True,
        helmholtz_synthesis_rank: int = 0,
        helmholtz_synthesis_late_rank: int = 0,
        helmholtz_synthesis_late_frequencies: int = 0,
        helmholtz_synthesis_frequency_softmax: bool = False,
        helmholtz_synthesis_source_onset_phase: bool = False,
        helmholtz_source_relative_coordinates: bool = False,
        helmholtz_spectral_bypass: bool = False,
        helmholtz_spectral_bypass_per_branch: bool = False,
        helmholtz_background_conditioning: bool = False,
        helmholtz_background_sigma_cells: float = 2.0,
        helmholtz_background_global_propagator: bool = False,
        helmholtz_background_propagation_modes: int = 48,
        helmholtz_background_direct_frequency_head: bool = False,
        helmholtz_background_direct_frequencies: int = 32,
        helmholtz_background_direct_spectral_experts: int = 0,
        adapter_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        if pyramid_levels <= 0 or saved_time_count <= 1:
            raise ValueError("local field pyramid levels and saved-time count must be positive")
        if domain_t_s <= 0 or domain_diagonal_m <= 0 or causal_width_s <= 0:
            raise ValueError("local field domain and causal scales must be positive")
        self.width = int(width)
        self.pyramid_levels = int(pyramid_levels)
        self.domain_t_s = float(domain_t_s)
        self.domain_diagonal_m = float(domain_diagonal_m)
        fallback_side = float(domain_diagonal_m) / math.sqrt(2.0)
        self.domain_x_m = fallback_side if domain_x_m is None else float(domain_x_m)
        self.domain_z_m = fallback_side if domain_z_m is None else float(domain_z_m)
        self.causal_width_s = float(causal_width_s)
        self.gabor_scales_s = tuple(float(value) for value in gabor_scales_s)

        base = self.width
        channels = tuple(base * int(m) for m in channel_multipliers)
        if channels[0] != base:
            raise ValueError("local field first channel multiplier must be 1")

        # --- record-level spatial conditioning (time-independent) ---
        self.multiscale_fuse = nn.Conv2d(base * self.pyramid_levels, base, kernel_size=1)
        self.source_projection = nn.Linear(base, base)
        self.map_projection = nn.Conv2d(base, base, kernel_size=1)
        # --- per-(record,time) conditioning ---
        self.time_mlp = nn.Sequential(
            nn.Linear(5, base), nn.GELU(), nn.Linear(base, 2 * base)
        )
        self.saved_time_embedding = nn.Embedding(saved_time_count, base)
        self.extended_late_features = bool(extended_late_features)
        phase_channels = 13 if self.extended_late_features else 12
        self.phase_projection = nn.Conv2d(phase_channels, base, kernel_size=1)
        if self.extended_late_features:
            # Zero-init the 13th (travel-progress) input weight so a warm-started
            # model reproduces its 12-channel prediction exactly at load, and the
            # new long-travel signal fades in only as training uses it.
            with torch.no_grad():
                self.phase_projection.weight[:, 12:].zero_()
        # --- translation-equivariant multi-scale generator ---
        # When the temporal operator reads the checkpointed U-Net output, the
        # non-reentrant saved-tensor hooks corrupt at epoch boundaries (pack_hook
        # assert).  Use reentrant checkpointing in that case — mathematically
        # identical, hook-mechanism robust.
        # The temporal operator AND the Helmholtz synthesis head both consume the U-Net
        # output OUTSIDE its checkpoint scope; with the outer full_forward_checkpointing
        # the two non-reentrant checkpoints corrupt the saved-tensor hooks at a backward
        # boundary (pack_hook assert).  Reentrant checkpointing is mathematically
        # identical and hook-mechanism robust in that case.
        self.unet = _UNet(
            channels,
            activation_checkpointing=activation_checkpointing,
            reentrant_checkpoint=int(temporal_operator_rank) > 0 or bool(helmholtz_synthesis),
        )
        self.output = nn.Conv2d(base, 1, kernel_size=1)
        self.temporal_operator = (
            None
            if int(temporal_operator_rank) <= 0
            else _ContinuousTimePropagationBasis(
                base,
                rank=int(temporal_operator_rank),
                spatial_kernel=int(temporal_operator_spatial_kernel),
                gate_init=float(adapter_gate_init),
            )
        )
        # Arrival-aligned characteristic warp (r4): repositions the rendered field
        # along grad(T) to correct wavefront position/phase.  Default off keeps every
        # existing run byte-identical; zero-init shift head keeps a warm-started
        # model an exact no-op at load.
        self.warp = (
            _EikonalArrivalWarp(
                base,
                max_shift_cells=float(warp_max_shift_cells),
                shift_dilation=int(warp_shift_dilation),
            )
            if bool(warp)
            else None
        )
        # Multi-arrival mixture warp (A4, class-A): ADDITIVE later-arrival paths applied
        # ON TOP of ``self.warp`` (Option B).  ``self.warp`` remains the parent-trained
        # path 0; A4 adds K delayed/displaced reflections.  Default off keeps every
        # existing run byte-identical; zero-init path gates keep a warm-started model an
        # exact no-op at load (starts EXACTLY at the loaded warp ceiling).
        self.multi_arrival = (
            _MultiArrivalWarp(
                base,
                num_paths=int(multi_arrival_paths),
                max_shift_cells=float(multi_arrival_max_shift_cells),
                max_delay_frac=float(multi_arrival_max_delay_frac),
                gate_init=float(adapter_gate_init),
            )
            if bool(multi_arrival)
            else None
        )
        # Dynamic Green/scattering kernel (r5): a radius-growing scattered-field
        # residual applied AFTER the arrival warp.  Default off keeps every existing
        # run byte-identical; zero-init gate keeps a warm-started model an exact no-op.
        self.green_kernel = (
            _DynamicGreenScatteringKernel(
                base,
                kernel_size=int(green_kernel_size),
                dilations=tuple(green_dilations),
            )
            if bool(green_kernel)
            else None
        )
        # Continuous temporal latent basis (A3, class-A coarse-field enrichment):
        # a query-independent M-anchor bank from the time-independent conditioning,
        # combined with continuous per-frame coefficients.  Default off keeps every
        # existing run byte-identical; zero-init gate keeps a warm-started model an
        # exact no-op.  A class-A enrichment of the DOMINANT layered+marmousi late
        # error (verified co-dominant ~79%), a parallel option to the class-B stack.
        self.temporal_latent = (
            _ContinuousTemporalLatentBasis(
                base,
                rank=int(temporal_latent_rank),
                harmonics=int(temporal_latent_harmonics),
                gate_init=float(adapter_gate_init),
            )
            if bool(temporal_latent_basis)
            else None
        )
        # Dispersive modal field (A5, class-A ESCALATION): per-pixel LEARNED-frequency
        # modal sum sum_m A_m cos(omega_m(x,z) t_rel + phi_m).  Default off keeps every
        # existing run byte-identical; zero-init gate keeps a warm-started model an exact
        # no-op.  The single structural factor over A3 is location-dependent dispersion
        # (learned per-pixel omega), the escalation A3's falsification gate points to.
        self.dispersive_modal = (
            _DispersiveModalField(
                base,
                modes=int(dispersive_modal_modes),
                max_frequency=float(dispersive_modal_max_frequency),
                gate_init=float(adapter_gate_init),
            )
            if bool(dispersive_modal)
            else None
        )
        # Windowed characteristic propagation (C1, contract-safe small-window space-time):
        # advects the already-warp-corrected field along grad(T) to a DETERMINISTIC
        # neighbor-time window (function of the saved index alone) and couples the
        # snapshots with a learned bilinear space-time operator.  The first module to
        # reduce over a genuine (deterministic) time axis -> ties wavefront position
        # across saved times (attacks the position/phase floor the per-frame-diagonal
        # modules cannot).  Default off keeps every existing run byte-identical; the
        # center tap (Delta_t=0) reproduces the field so gate_init=0 is an exact no-op.
        self.windowed_propagation = (
            _WindowedCharacteristicPropagation(
                base,
                window=int(windowed_propagation_window),
                stride=int(windowed_propagation_stride),
                rank=int(windowed_propagation_rank),
                max_advect_cells=float(windowed_propagation_max_advect_cells),
                gate_init=float(adapter_gate_init),
            )
            if bool(windowed_propagation)
            else None
        )
        # Temporal-frequency (Helmholtz) synthesis (H1): a PRIMARY coarse-field form
        # (not a gated residual).  When on, the U-Net renders ONCE per record from the
        # time-independent conditioning; a complex-field head predicts ``nf`` static
        # Helmholtz fields and the FIXED inverse temporal transform synthesizes every
        # requested saved-time frame.  This changes the time REPRESENTATION (analytic
        # Fourier basis) instead of re-placing the wavefront per frame -> attacks the
        # position/phase floor the diagonal per-frame forms cannot.  Query-invariant by
        # construction.  Mutually exclusive with the per-frame residual/warp stack.
        self.helmholtz_synthesis = (
            _HelmholtzSynthesisField(
                base,
                num_frequencies=int(helmholtz_synthesis_frequencies),
                wkb_phase=bool(helmholtz_synthesis_wkb_phase),
                rank=int(helmholtz_synthesis_rank),
                late_rank=int(helmholtz_synthesis_late_rank),
                late_frequencies=int(helmholtz_synthesis_late_frequencies),
                frequency_softmax=bool(helmholtz_synthesis_frequency_softmax),
            )
            if bool(helmholtz_synthesis)
            else None
        )
        self.helmholtz_source_onset_phase = bool(
            helmholtz_synthesis_source_onset_phase
        )
        self.helmholtz_source_relative_projection = (
            nn.Conv2d(3, base, kernel_size=1)
            if bool(helmholtz_source_relative_coordinates)
            else None
        )
        if self.helmholtz_source_relative_projection is not None:
            nn.init.zeros_(self.helmholtz_source_relative_projection.weight)
            nn.init.zeros_(self.helmholtz_source_relative_projection.bias)
        # Complete Fourier-coefficient supervision already learns a causal waveform.
        # Multiplying it by a pixel-dependent first-arrival sigmoid after synthesis
        # changes the fixed Fourier design and makes late-arrival pixels rank deficient.
        # Default True preserves every historical checkpoint; registered coefficient-
        # supervised runs may explicitly disable this non-parametric post-processing.
        self.helmholtz_apply_causal_gate = True
        # Multi-scale spectral bypass (VERDICT sec 27): test whether retaining more of
        # the high-rank render input can reduce the late scattering residual.  The small
        # positive gate deliberately gives the branch a usable gradient from step zero;
        # per-branch mode uses frequency-decaying positive gates.
        self.helmholtz_spectral_bypass = (
            _MultiScaleSpectralBypass(base, per_branch_gate=bool(helmholtz_spectral_bypass_per_branch))
            if bool(helmholtz_synthesis) and bool(helmholtz_spectral_bypass)
            else None
        )
        self.helmholtz_background_conditioner = (
            _BackgroundBornConditioner(
                base,
                num_frequencies=int(helmholtz_synthesis_frequencies),
                background_sigma_cells=float(helmholtz_background_sigma_cells),
                global_propagation=bool(helmholtz_background_global_propagator),
                propagation_modes=int(helmholtz_background_propagation_modes),
                direct_frequency_output=bool(
                    helmholtz_background_direct_frequency_head
                ),
                direct_frequency_count=int(
                    helmholtz_background_direct_frequencies
                ),
                direct_spectral_experts=int(
                    helmholtz_background_direct_spectral_experts
                ),
            )
            if bool(helmholtz_synthesis) and bool(helmholtz_background_conditioning)
            else None
        )
        if self.residual:
            # ControlNet-style zero convolution: the local field contributes
            # exactly nothing at initialisation, so the model starts on the
            # global-coarse (MIONet) trajectory and the U-Net fades in as this
            # projection learns off zero. Keeps the residual variant from paying
            # the from-scratch warm-up the replace variant pays.
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        velocity_mps: torch.Tensor,
        medium: MediumEncoding,
        source: SourceEncoding,
        source_parameters: torch.Tensor,
        record_to_medium: torch.Tensor,
        time_s: torch.Tensor,
        travel: RayTravelTime,
        saved_time_indices: torch.Tensor,
        saved_time_values: torch.Tensor | None = None,
        background_normalized: torch.Tensor | None = None,
        background_conditioning_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=source.hidden.device)
        if time_s.ndim != 2:
            raise ValueError("local field requested times must be [record,time]")
        records, count = time_s.shape
        full_size = medium.pyramid[0].shape[-2:]
        height, width = int(full_size[0]), int(full_size[1])
        if mapping.shape != (records,) or source_parameters.shape != (records, 5):
            raise ValueError("local field source/medium mapping does not match requested times")
        if saved_time_indices.shape != (records, count):
            raise ValueError("saved time indices must match requested times")

        # medium pyramid features resampled to full resolution + source conditioning
        levels = [medium.pyramid[0][mapping]]
        levels.extend(
            F.interpolate(level[mapping], size=full_size, mode="bilinear", align_corners=True)
            for level in medium.pyramid[1:]
        )
        spatial = (
            self.multiscale_fuse(torch.cat(levels, dim=1))
            + self.source_projection(source.hidden)[:, :, None, None]
            + self.map_projection(source.map_field)
        )

        seconds = torch.as_tensor(travel.seconds, dtype=torch.float32, device=spatial.device)
        arrival = seconds.reshape(records, height, width)

        if self.helmholtz_synthesis is not None:
            # H1: render ONCE per record from the time-INDEPENDENT conditioning plus the
            # static eikonal phase anchor; predict nf complex Helmholtz fields and
            # synthesize every requested saved-time frame with the fixed inverse
            # transform.  No per-frame FiLM / saved-index embedding -> the time axis is
            # carried analytically, not re-guessed per frame.
            if saved_time_values is None:
                raise ValueError(
                    "helmholtz_synthesis requires saved_time_values (the stored-time axis)"
                )
            onset_s = source_parameters[:, 3][:, None, None]
            phase_arrival = (
                arrival + onset_s
                if self.helmholtz_source_onset_phase
                else arrival
            )
            # Keep the learned spatial phase anchor on propagation travel only.
            # The new variable is solely the analytic source-time shift in the
            # Fourier synthesis/target pair, so all record features stay unchanged.
            tau_norm = (arrival / self.domain_t_s)[:, None]        # (R,1,H,W)
            record_input = spatial + self.helmholtz_synthesis.phase_anchor(tau_norm)
            if self.helmholtz_source_relative_projection is not None:
                x_m = torch.linspace(
                    0.0, self.domain_x_m, width,
                    device=spatial.device, dtype=spatial.dtype,
                )
                z_m = torch.linspace(
                    0.0, self.domain_z_m, height,
                    device=spatial.device, dtype=spatial.dtype,
                )
                dx = (
                    x_m[None, None, :]
                    - source_parameters[:, 0].to(spatial.dtype)[:, None, None]
                ) / self.domain_diagonal_m
                dz = (
                    z_m[None, :, None]
                    - source_parameters[:, 1].to(spatial.dtype)[:, None, None]
                ) / self.domain_diagonal_m
                dx = dx.expand(-1, height, -1)
                dz = dz.expand(-1, -1, width)
                relative = torch.stack(
                    (dx, dz, torch.sqrt(dx.square() + dz.square())), dim=1
                )
                record_input = record_input + self.helmholtz_source_relative_projection(
                    relative
                )
            direct_frequency_head = False
            if self.helmholtz_background_conditioner is not None:
                if background_conditioning_features is None and background_normalized is None:
                    raise ValueError(
                        "helmholtz background conditioning requires the complete "
                        "normalized P_bg stored-time field"
                    )
                if background_conditioning_features is None:
                    omega = self.helmholtz_synthesis.frequency_bank(saved_time_values)
                    background_conditioning_features = (
                        self.helmholtz_background_conditioner(
                            background_normalized,
                            velocity_mps,
                            mapping,
                            arrival,
                            omega,
                        )
                    )
                direct_frequency_head = bool(
                    self.helmholtz_background_conditioner.direct_frequency_output
                )
                if (
                    not direct_frequency_head
                    and background_conditioning_features.shape != spatial.shape
                ):
                    raise ValueError(
                        "precomputed background conditioning features do not match "
                        "the record-level spatial conditioning"
                    )
                if not self.helmholtz_background_conditioner.global_propagation:
                    record_input = record_input + background_conditioning_features
            rendered_record = self.unet(record_input)             # (R, width, H, W)
            late_record_gate = None
            if (
                self.helmholtz_background_conditioner is not None
                and self.helmholtz_background_conditioner.global_propagation
            ):
                if not direct_frequency_head:
                    rendered_record = rendered_record + background_conditioning_features
                if self.helmholtz_synthesis.late_rank > 0:
                    # A trainable late-rank head otherwise acts on the frozen parent
                    # render even for a constant medium and can invent scattering in
                    # the uniform family.  Gate only the new late contribution by the
                    # exact slowness-contrast invariant; the transferred rank-8 field
                    # remains untouched.
                    contrast = self.helmholtz_background_conditioner.slowness_contrast(
                        velocity_mps[mapping].float()
                    )
                    late_record_gate = (
                        contrast.abs().flatten(1).amax(dim=1) >= 1.0e-5
                    ).to(rendered_record.dtype)
            if self.helmholtz_spectral_bypass is not None:
                # add high-rank spectral structure straight from the high-rank input
                rendered_record = rendered_record + self.helmholtz_spectral_bypass(record_input)
            field = self.helmholtz_synthesis(
                rendered_record,
                phase_arrival,
                time_s,
                saved_time_values,
                domain_t_s=self.domain_t_s,
                late_record_gate=late_record_gate,
            )
            direct_frequency_field = None
            if direct_frequency_head:
                direct_frequency_field = self.helmholtz_background_conditioner.synthesize_direct_frequency_output(
                    background_conditioning_features,
                    time_s,
                    saved_time_values,
                )
            onset = source_parameters[:, 3][:, None, None, None]
            tau = time_s[:, :, None, None] - onset - arrival[:, None]
            gate = torch.sigmoid(tau / self.causal_width_s)
            if direct_frequency_field is not None:
                # The direct Fourier branch represents the complete scattered
                # waveform on an absolute-time basis.  Applying the first-arrival
                # gate after synthesis changes the otherwise fixed, full-rank
                # Fourier design independently at every pixel and makes late-arrival
                # pixels rank deficient.  Keep the transferred parent causal while
                # letting supervision learn the scattering branch's causality.
                return field * gate + direct_frequency_field
            if not bool(self.helmholtz_apply_causal_gate):
                return field
            return field * gate

        # per-frame FiLM from continuous-time + saved-index embedding (mirrors decoder)
        relative_time = time_s - source_parameters[:, None, 3]
        frequency = source_parameters[:, None, 2].expand_as(time_s)
        global_phase = 2.0 * math.pi * frequency * relative_time
        time_features = torch.stack(
            (
                time_s / self.domain_t_s,
                relative_time / self.domain_t_s,
                frequency / 50.0,
                torch.sin(global_phase),
                torch.cos(global_phase),
            ),
            dim=-1,
        )
        scale, bias = self.time_mlp(time_features).chunk(2, dim=-1)
        conditioned = (
            spatial[:, None] * (1.0 + scale[:, :, :, None, None])
            + bias[:, :, :, None, None]
            + self.saved_time_embedding(saved_time_indices)[:, :, :, None, None]
        )
        flat = conditioned.reshape(records * count, self.width, height, width)

        # physically-correct wavefront location/phase as dense input channels
        phase = dense_propagation_features(
            travel,
            time_s,
            source_parameters,
            height=height,
            width=width,
            domain_t_s=self.domain_t_s,
            domain_diagonal_m=self.domain_diagonal_m,
            gabor_scales_s=self.gabor_scales_s,
            include_travel_progress=self.extended_late_features,
        )
        phase_channels = 13 if self.extended_late_features else 12
        flat = flat + self.phase_projection(
            phase.reshape(records * count, phase_channels, height, width)
        )

        rendered = self.unet(flat)
        field = self.output(rendered).reshape(records, count, height, width)
        if self.temporal_latent is not None:
            # A3 (class-A): continuous temporal latent-basis enrichment of the coarse
            # field, built from the time-INDEPENDENT record conditioning ``spatial``
            # (query-invariant; zero-init gate => exact warm-start no-op).
            field = field + self.temporal_latent(
                spatial,
                time_s,
                source_parameters,
                domain_t_s=self.domain_t_s,
            )
        if self.dispersive_modal is not None:
            # A5 (class-A escalation): per-pixel learned-dispersion modal enrichment of
            # the coarse field, from the time-INDEPENDENT record conditioning ``spatial``
            # (query-invariant; zero-init gate => exact warm-start no-op).
            field = field + self.dispersive_modal(
                spatial,
                time_s,
                source_parameters,
                domain_t_s=self.domain_t_s,
            )
        if self.temporal_operator is not None:
            # Query-invariant continuous-time propagation residual (zero-init gate).
            field = field + self.temporal_operator(
                rendered,
                time_s,
                source_parameters,
                records=records,
                count=count,
                domain_t_s=self.domain_t_s,
            )

        # structural causal gate: p approximately 0 before eikonal first arrival.
        # The eikonal travel-time field is shared by the arrival warp and the gate.
        seconds = torch.as_tensor(travel.seconds, dtype=torch.float32, device=field.device)
        arrival = seconds.reshape(records, height, width)

        if self.warp is not None:
            # r4: reposition the field along the propagation characteristic grad(T)
            # to correct wavefront position/phase (exact no-op until the warp learns).
            field = self.warp(field, rendered, arrival)
        if self.multi_arrival is not None:
            # A4: add delayed, displaced LATER arrivals on top of the path-0 warp
            # (Option B: self.warp is the trained path 0; extra paths are zero-gated
            # at warm-start so this is an exact no-op that starts at the warp ceiling).
            onset_t = source_parameters[:, 3][:, None, None, None]
            field = self.multi_arrival(
                field,
                rendered,
                arrival,
                spatial,
                time_s,
                onset_t,
                self.causal_width_s,
                self.domain_t_s,
            )
        if self.windowed_propagation is not None:
            # C1: advect the (warp-corrected) field along grad(T) to a deterministic
            # neighbor-time window and couple the snapshots -> ties wavefront position
            # across saved times (query-invariant; gate_init=0 => exact no-op).
            if saved_time_values is None:
                raise ValueError(
                    "windowed_propagation requires saved_time_values (the stored-time axis)"
                )
            field = self.windowed_propagation(
                field,
                rendered,
                arrival,
                time_s,
                saved_time_indices,
                saved_time_values,
                source_parameters,
                domain_t_s=self.domain_t_s,
            )

        onset = source_parameters[:, 3][:, None, None, None]
        tau = time_s[:, :, None, None] - onset - arrival[:, None]
        if self.green_kernel is not None:
            # r5: scattered-field residual whose support radius grows with elapsed
            # post-arrival progress (clamped >= 0 so pre-arrival pixels get no wake).
            progress = torch.clamp(tau, min=0.0) / self.domain_t_s
            field = field + self.green_kernel(
                field, rendered, progress, records=records, count=count
            )
        gate = torch.sigmoid(tau / self.causal_width_s)
        return field * gate


__all__ = ["LocalPropagationFieldGenerator"]
