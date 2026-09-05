"""Precision-first LoRA and guarded QLoRA-inspired reference layers.

Full-precision LoRA is the deployment baseline.  The quantized classes are a
portable numerical reference for train-only accuracy experiments; INT4 values
are stored in int8 containers and are not claimed to provide CUDA packing or
speedup.  A quantized candidate must pass an output-equivalence guard or the
full-precision result is returned unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    group_size: int
    bits: int

    def dequantize(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        rows = self.values.shape[0]
        restored = self.values.float() * self.scales
        restored = restored.reshape(rows, -1)[:, : math.prod(self.original_shape[1:])]
        restored = restored.reshape(self.original_shape)
        return restored if dtype is None else restored.to(dtype=dtype)


@dataclass(frozen=True)
class PrecisionGuardResult:
    output: torch.Tensor
    accepted_quantized: bool
    relative_error: float
    tolerance: float


def groupwise_symmetric_quantize(
    weight: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 64,
) -> QuantizedTensor:
    """Quantize each output row in contiguous groups using symmetric integers."""
    value = torch.as_tensor(weight).detach().float()
    if value.ndim < 2 or bits not in (2, 3, 4, 8) or group_size <= 0:
        raise ValueError("quantization expects a matrix-like weight, 2/3/4/8 bits, and positive groups")
    rows = value.shape[0]
    flat = value.reshape(rows, -1)
    groups = math.ceil(flat.shape[1] / int(group_size))
    padded_size = groups * int(group_size)
    if padded_size != flat.shape[1]:
        flat = F.pad(flat, (0, padded_size - flat.shape[1]))
    grouped = flat.reshape(rows, groups, int(group_size))
    qmax = float(2 ** (int(bits) - 1) - 1)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / qmax
    quantized = torch.round(grouped / scales).clamp(-qmax, qmax).to(torch.int8)
    return QuantizedTensor(
        values=quantized,
        scales=scales,
        original_shape=tuple(value.shape),
        group_size=int(group_size),
        bits=int(bits),
    )


def _svd_error_factors(
    error: torch.Tensor,
    *,
    rank: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = error.reshape(error.shape[0], -1).float()
    retained = min(int(rank), *matrix.shape)
    if retained <= 0 or scale <= 0.0:
        raise ValueError("compensation rank and LoRA scale must be positive")
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    root = singular[:retained].clamp_min(0.0).sqrt()
    up = left[:, :retained] * root[None] / math.sqrt(scale)
    down = root[:, None] * right[:retained] / math.sqrt(scale)
    return up, down


class PrecisionFirstLoRALinear(nn.Module):
    """Frozen full-precision linear layer plus a zero-initialized LoRA/DoRA update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float | None = None,
        train_magnitude: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))
        self.magnitude_delta = (
            nn.Parameter(torch.zeros(base.out_features)) if train_magnitude else None
        )

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        if self.magnitude_delta is not None:
            base = base * (1.0 + self.magnitude_delta)
        update = F.linear(F.linear(value, self.lora_a), self.lora_b)
        return base + self.scale * update

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        values = [self.lora_a, self.lora_b]
        if self.magnitude_delta is not None:
            values.append(self.magnitude_delta)
        return tuple(values)

    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        multiplier = (
            torch.ones(self.base.out_features, device=self.base.weight.device, dtype=self.base.weight.dtype)
            if self.magnitude_delta is None
            else 1.0 + self.magnitude_delta.to(self.base.weight)
        )
        weight = self.base.weight * multiplier[:, None]
        weight = weight + self.scale * (self.lora_b @ self.lora_a).to(weight)
        bias = None if self.base.bias is None else self.base.bias * multiplier
        return weight, bias


class PrecisionFirstLoRAConv2d(nn.Module):
    """Full-precision convolution with an exact-zero low-rank convolutional update."""

    def __init__(
        self,
        base: nn.Conv2d,
        *,
        rank: int = 8,
        alpha: float | None = None,
        train_magnitude: bool = False,
    ) -> None:
        super().__init__()
        if base.groups != 1 or base.padding_mode != "zeros":
            raise ValueError("reference LoRA convolution currently requires groups=1 and zero padding")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.lora_down = nn.Conv2d(
            base.in_channels,
            rank,
            base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_up.weight)
        self.magnitude_delta = (
            nn.Parameter(torch.zeros(base.out_channels)) if train_magnitude else None
        )

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        if self.magnitude_delta is not None:
            base = base * (1.0 + self.magnitude_delta[None, :, None, None])
        return base + self.scale * self.lora_up(self.lora_down(value))

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        values = [self.lora_down.weight, self.lora_up.weight]
        if self.magnitude_delta is not None:
            values.append(self.magnitude_delta)
        return tuple(values)

    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        multiplier = (
            torch.ones(self.base.out_channels, device=self.base.weight.device, dtype=self.base.weight.dtype)
            if self.magnitude_delta is None
            else 1.0 + self.magnitude_delta.to(self.base.weight)
        )
        up = self.lora_up.weight[:, :, 0, 0]
        down = self.lora_down.weight
        update = torch.einsum("or,rihw->oihw", up, down)
        weight = self.base.weight * multiplier[:, None, None, None]
        weight = weight + self.scale * update.to(weight)
        bias = None if self.base.bias is None else self.base.bias * multiplier
        return weight, bias


class QuantizedLoRALinearCandidate(nn.Module):
    """Groupwise-quantized frozen linear layer with optional SVD error compensation."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float | None = None,
        bits: int = 4,
        group_size: int = 64,
        compensate: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("quantized LoRA rank must be positive")
        quantized = groupwise_symmetric_quantize(
            base.weight, bits=bits, group_size=group_size
        )
        self.register_buffer("qweight", quantized.values)
        self.register_buffer("qscale", quantized.scales)
        self.original_shape = quantized.original_shape
        self.group_size = quantized.group_size
        self.bits = quantized.bits
        self.register_buffer(
            "bias",
            None if base.bias is None else base.bias.detach().clone(),
        )
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        if compensate:
            dequantized = self.dequantized_weight(dtype=base.weight.dtype)
            up, down = _svd_error_factors(
                base.weight.detach() - dequantized,
                rank=rank,
                scale=self.scale,
            )
            retained = down.shape[0]
            self.lora_b.data[:, :retained].copy_(up.to(self.lora_b))
            self.lora_a.data[:retained].copy_(down.to(self.lora_a))

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def dequantized_weight(self, *, dtype: torch.dtype) -> torch.Tensor:
        quantized = QuantizedTensor(
            self.qweight,
            self.qscale,
            self.original_shape,
            self.group_size,
            self.bits,
        )
        return quantized.dequantize(dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight(dtype=value.dtype)
        base = F.linear(value, weight, None if self.bias is None else self.bias.to(value))
        update = F.linear(F.linear(value, self.lora_a.to(value)), self.lora_b.to(value))
        return base + self.scale * update


class QuantizedLoRAConv2dCandidate(nn.Module):
    """Portable quantized Conv2d candidate with LoftQ-style SVD compensation."""

    def __init__(
        self,
        base: nn.Conv2d,
        *,
        rank: int = 8,
        alpha: float | None = None,
        bits: int = 4,
        group_size: int = 64,
        compensate: bool = True,
    ) -> None:
        super().__init__()
        if base.groups != 1 or base.padding_mode != "zeros" or rank <= 0:
            raise ValueError("quantized LoRA Conv2d requires groups=1, zero padding, and positive rank")
        quantized = groupwise_symmetric_quantize(
            base.weight, bits=bits, group_size=group_size
        )
        self.register_buffer("qweight", quantized.values)
        self.register_buffer("qscale", quantized.scales)
        self.original_shape = quantized.original_shape
        self.group_size = quantized.group_size
        self.bits = quantized.bits
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.register_buffer(
            "bias",
            None if base.bias is None else base.bias.detach().clone(),
        )
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.lora_down = nn.Parameter(torch.zeros(rank, *base.weight.shape[1:]))
        self.lora_up = nn.Parameter(torch.zeros(base.out_channels, rank))
        if compensate:
            dequantized = self.dequantized_weight(dtype=base.weight.dtype)
            up, down = _svd_error_factors(
                base.weight.detach() - dequantized,
                rank=rank,
                scale=self.scale,
            )
            retained = down.shape[0]
            self.lora_up.data[:, :retained].copy_(up.to(self.lora_up))
            self.lora_down.data[:retained].copy_(down.reshape(retained, *base.weight.shape[1:]).to(self.lora_down))

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def dequantized_weight(self, *, dtype: torch.dtype) -> torch.Tensor:
        return QuantizedTensor(
            self.qweight,
            self.qscale,
            self.original_shape,
            self.group_size,
            self.bits,
        ).dequantize(dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = F.conv2d(
            value,
            self.dequantized_weight(dtype=value.dtype),
            None if self.bias is None else self.bias.to(value),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        down = F.conv2d(
            value,
            self.lora_down.to(value),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        update = torch.einsum("or,brhw->bohw", self.lora_up.to(value), down)
        return base + self.scale * update


PROTECTED_QUANTIZATION_TOKENS = (
    "spatial",
    "frequency",
    "wfp",
    "spectral",
    "fft",
    "phase",
    "travel",
    "norm",
    "physical_head",
    "cpml_head",
)


def quantization_policy(name: str, module: nn.Module) -> str:
    """Return a conservative policy for the current acoustic operator."""
    lowered = str(name).lower()
    if any(token in lowered for token in PROTECTED_QUANTIZATION_TOKENS):
        return "full_precision_protected"
    if isinstance(module, nn.Linear):
        return "quantized_candidate"
    if isinstance(module, nn.Conv2d) and module.groups == 1 and module.kernel_size == (1, 1):
        return "quantized_candidate"
    return "full_precision_default"


@torch.inference_mode()
def guarded_quantized_forward(
    reference: nn.Module,
    candidate: nn.Module,
    value: torch.Tensor,
    *,
    tolerance: float,
) -> PrecisionGuardResult:
    """Use a quantized output only when it is sufficiently close to its FP reference."""
    if tolerance < 0.0:
        raise ValueError("precision tolerance must be nonnegative")
    reference_output = reference(value)
    candidate_output = candidate(value)
    relative = float(
        (candidate_output.float() - reference_output.float()).norm()
        / reference_output.float().norm().clamp_min(1.0e-12)
    )
    accepted = math.isfinite(relative) and relative <= float(tolerance)
    return PrecisionGuardResult(
        output=candidate_output if accepted else reference_output,
        accepted_quantized=accepted,
        relative_error=relative,
        tolerance=float(tolerance),
    )


__all__ = [
    "PROTECTED_QUANTIZATION_TOKENS",
    "PrecisionFirstLoRAConv2d",
    "PrecisionFirstLoRALinear",
    "PrecisionGuardResult",
    "QuantizedLoRAConv2dCandidate",
    "QuantizedLoRALinearCandidate",
    "QuantizedTensor",
    "groupwise_symmetric_quantize",
    "guarded_quantized_forward",
    "quantization_policy",
]
