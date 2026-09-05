"""Fourier operators for traces sampled at nonuniform physical times."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
from torch import nn


def _validate_time_vector(time_s: torch.Tensor) -> None:
    if time_s.ndim != 1:
        raise ValueError("time_s must be one-dimensional")
    if time_s.numel() < 2:
        raise ValueError("time_s must contain at least two samples")
    if not time_s.is_floating_point():
        raise ValueError("time_s must have a floating dtype")
    if not bool(torch.isfinite(time_s).all()):
        raise ValueError("time_s must contain only finite values")
    if not bool(torch.all(torch.diff(time_s) > 0)):
        raise ValueError("time_s must be strictly increasing")


class _ValidatedTimeGrid:
    """Capability for an owned, strictly validated physical-time snapshot."""

    __slots__ = (
        "_time_s",
        "_identity",
        "_data_ptr",
        "_shape",
        "_dtype",
        "_device",
        "_version",
    )

    def __init__(self, time_s: torch.Tensor) -> None:
        if not isinstance(time_s, torch.Tensor):
            raise ValueError("time_s must be a tensor")
        _validate_time_vector(time_s)
        snapshot = time_s.clone(memory_format=torch.contiguous_format)
        self._time_s = snapshot
        self._identity = id(snapshot)
        self._data_ptr = snapshot.data_ptr()
        self._shape = snapshot.shape
        self._dtype = snapshot.dtype
        self._device = snapshot.device
        self._version = snapshot._version

    @property
    def time_s(self) -> torch.Tensor:
        return self._time_s.clone(memory_format=torch.contiguous_format)

    def __reduce__(self) -> tuple[object, tuple[torch.Tensor]]:
        self.assert_current()
        return (_rebuild_validated_time_grid, (self._time_s,))

    def __deepcopy__(self, memo: dict[int, object]) -> "_ValidatedTimeGrid":
        self.assert_current()
        restored = _ValidatedTimeGrid(self._time_s)
        memo[id(self)] = restored
        return restored

    def assert_current(self) -> None:
        time_s = self._time_s
        current = (
            id(time_s),
            time_s.data_ptr(),
            time_s.shape,
            time_s.dtype,
            time_s.device,
            time_s._version,
        )
        expected = (
            self._identity,
            self._data_ptr,
            self._shape,
            self._dtype,
            self._device,
            self._version,
        )
        if current != expected or not time_s.is_contiguous():
            raise ValueError("validated time grid snapshot was mutated")


def _rebuild_validated_time_grid(time_s: torch.Tensor) -> _ValidatedTimeGrid:
    return _ValidatedTimeGrid(time_s)


def _make_validated_time_grid(time_s: torch.Tensor) -> _ValidatedTimeGrid:
    return _ValidatedTimeGrid(time_s)


def _require_validated_time_grid(
    time_grid: _ValidatedTimeGrid,
) -> torch.Tensor:
    if not isinstance(time_grid, _ValidatedTimeGrid):
        raise TypeError("expected a validated time grid capability")
    time_grid.assert_current()
    return time_grid._time_s


def _trapezoid_weights_validated(time_grid: _ValidatedTimeGrid) -> torch.Tensor:
    time_s = _require_validated_time_grid(time_grid)
    intervals = torch.diff(time_s)
    weights = torch.empty_like(time_s)
    weights[0] = intervals[0] / 2
    weights[-1] = intervals[-1] / 2
    weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2
    return weights


def trapezoid_weights(time_s: torch.Tensor) -> torch.Tensor:
    """Return physical-time trapezoid weights for a strictly increasing grid."""

    return _trapezoid_weights_validated(_make_validated_time_grid(time_s))


def nonuniform_fourier_analysis(
    x: torch.Tensor,
    time_s: torch.Tensor,
    modes: int,
) -> torch.Tensor:
    """Project ``x[B,Q,T,C]`` onto normalized physical-time Fourier modes."""

    if x.ndim != 4:
        raise ValueError("expected x shaped [B,Q,T,C]")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("x must have a real floating dtype")
    if time_s.ndim != 1 or time_s.numel() != x.shape[2]:
        raise ValueError("time_s must be a one-dimensional vector matching x time")
    if not isinstance(modes, int) or isinstance(modes, bool) or not 1 <= modes <= x.shape[2]:
        raise ValueError("modes must be an integer between 1 and the time length")

    return _nonuniform_fourier_analysis_validated(
        x, _make_validated_time_grid(time_s), modes
    )


def _nonuniform_fourier_analysis_validated(
    x: torch.Tensor,
    time_grid: _ValidatedTimeGrid,
    modes: int,
) -> torch.Tensor:
    time_s = _require_validated_time_grid(time_grid)
    if x.ndim != 4 or time_s.numel() != x.shape[2]:
        raise ValueError("time_s must match x time")

    real_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    complex_dtype = (
        torch.complex128 if real_dtype == torch.float64 else torch.complex64
    )
    time = time_s.to(device=x.device, dtype=torch.float64)
    tau = (time - time[0]) / (time[-1] - time[0])
    frequencies = 2.0 * torch.pi * torch.arange(
        modes, device=x.device, dtype=torch.float64
    )
    phase = (tau[:, None] * frequencies[None, :]).to(real_dtype)
    basis = torch.complex(torch.cos(phase), -torch.sin(phase))
    intervals = torch.diff(time)
    weights = torch.empty_like(time)
    weights[0] = intervals[0] / 2
    weights[-1] = intervals[-1] / 2
    weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2
    normalized_weights = (weights / weights.sum()).to(complex_dtype)
    return torch.einsum(
        "bqtc,tm,t->bqmc", x.to(complex_dtype), basis, normalized_weights
    )


def nonuniform_fourier_synthesis(
    coefficients: torch.Tensor,
    time_s: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Fourier coefficients on ``time_s`` and return their real part."""

    if coefficients.ndim != 4:
        raise ValueError("expected coefficients shaped [B,Q,M,C]")
    if coefficients.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("coefficients must have dtype complex64 or complex128")
    if time_s.ndim != 1:
        raise ValueError("time_s must be a one-dimensional time vector")
    if coefficients.shape[2] < 1:
        raise ValueError("coefficients must contain at least one mode")

    return _nonuniform_fourier_synthesis_validated(
        coefficients, _make_validated_time_grid(time_s)
    )


def _nonuniform_fourier_synthesis_validated(
    coefficients: torch.Tensor,
    time_grid: _ValidatedTimeGrid,
) -> torch.Tensor:
    time_s = _require_validated_time_grid(time_grid)

    time = time_s.to(device=coefficients.device, dtype=torch.float64)
    tau = (time - time[0]) / (time[-1] - time[0])
    frequencies = 2.0 * torch.pi * torch.arange(
        coefficients.shape[2], device=coefficients.device, dtype=torch.float64
    )
    real_dtype = (
        torch.float64 if coefficients.dtype == torch.complex128 else torch.float32
    )
    phase = (tau[:, None] * frequencies[None, :]).to(real_dtype)
    basis = torch.complex(torch.cos(phase), torch.sin(phase)).to(coefficients.dtype)
    dc = coefficients[:, :, :1, :].real
    if coefficients.shape[2] == 1:
        return dc.expand(-1, -1, time.numel(), -1)
    positive = torch.einsum(
        "bqmc,tm->bqtc", coefficients[:, :, 1:, :], basis[:, 1:]
    ).real
    return dc + 2.0 * positive


class NonUniformTemporalOperator(nn.Module):
    """A zero-initialized spectral correction over a complete 160-sample trace."""

    def __init__(
        self, channels: int, modes: int, residual_init: str = "identity"
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if not isinstance(modes, int) or isinstance(modes, bool) or not 1 <= modes <= 160:
            raise ValueError("modes must be an integer between 1 and 160")
        if residual_init != "identity":
            raise ValueError("residual_init must be 'identity'")

        self.channels = channels
        self.modes = modes
        self.dc_weight = nn.Parameter(torch.zeros(channels, channels, dtype=torch.float32))
        self.weight = nn.Parameter(
            torch.zeros(channels, channels, modes - 1, dtype=torch.complex64)
        )

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> "NonUniformTemporalOperator":
        real_probe = torch.empty(
            (), dtype=self.dc_weight.dtype, device=self.dc_weight.device
        )
        complex_probe = torch.empty(
            (), dtype=self.weight.dtype, device=self.weight.device
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Casting complex values to real discards the imaginary part.*",
                category=UserWarning,
            )
            converted_real = fn(real_probe)
            fn(complex_probe)

        if converted_real.dtype == torch.float64:
            real_dtype = torch.float64
            complex_dtype = torch.complex128
        else:
            real_dtype = torch.float32
            complex_dtype = torch.complex64
        target_device = converted_real.device

        def convert_parameter(tensor: torch.Tensor) -> torch.Tensor:
            dtype = complex_dtype if tensor.is_complex() else real_dtype
            return tensor.to(device=target_device, dtype=dtype)

        return super()._apply(convert_parameter, recurse=recurse)

    def forward(
        self,
        x: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[2] != 160 or x.shape[-1] != self.channels:
            raise ValueError("expected x shaped [B,Q,160,C]")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("x must have a real floating dtype")

        if time_s.ndim == 1:
            time = time_s
        elif time_s.ndim == 2:
            if time_s.shape != (x.shape[0], 160):
                raise ValueError("batched time_s must have shape [B,160]")
            if not torch.equal(time_s, time_s[0].expand_as(time_s)):
                raise ValueError("batched time_s must contain one shared time vector")
            time = time_s[0]
        else:
            raise ValueError("time_s must have shape [160] or [B,160]")

        if time.numel() != 160:
            raise ValueError("expected 160 strictly increasing physical times")
        return self._forward_validated(x, _make_validated_time_grid(time))

    def _forward_validated(
        self,
        x: torch.Tensor,
        time_grid: _ValidatedTimeGrid,
    ) -> torch.Tensor:
        time = _require_validated_time_grid(time_grid)
        if x.ndim != 4 or x.shape[2] != 160 or x.shape[-1] != self.channels:
            raise ValueError("expected x shaped [B,Q,160,C]")
        if time.numel() != 160:
            raise ValueError("expected 160 strictly increasing physical times")

        output_dtype = x.dtype
        real_dtype = (
            torch.float64
            if x.dtype == torch.float64 or self.dc_weight.dtype == torch.float64
            else torch.float32
        )
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )
        x_work = x.to(real_dtype)
        coefficients = _nonuniform_fourier_analysis_validated(
            x_work, time_grid, self.modes
        )
        mixed_dc = torch.einsum(
            "bqi,ci->bqc",
            coefficients[:, :, 0, :].real,
            self.dc_weight.to(real_dtype),
        ).to(complex_dtype)[:, :, None, :]
        if self.modes == 1:
            mixed = mixed_dc
        else:
            mixed_positive = torch.einsum(
                "bqmi,cim->bqmc",
                coefficients[:, :, 1:, :],
                self.weight.to(complex_dtype),
            )
            mixed = torch.cat((mixed_dc, mixed_positive), dim=2)
        correction = _nonuniform_fourier_synthesis_validated(mixed, time_grid)
        return (x_work + correction).to(output_dtype)
