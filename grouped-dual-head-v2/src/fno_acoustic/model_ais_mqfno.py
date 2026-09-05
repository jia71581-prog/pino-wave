"""Adaptive importance-sampled multiscale full-trace query FNO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .ais_model_components import (
    DispersionResidualHead,
    LocalQueryEncoder,
    MultiScaleLocalQueryEncoder,
)
from .model_factorized import SpatialEncoder
from .temporal_operator import (
    NonUniformTemporalOperator,
    _ValidatedTimeGrid,
    _make_validated_time_grid,
    _require_validated_time_grid,
)


def _positive_integer(name: str, value: object, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _shared_full160_time(time_s: torch.Tensor, batch: int) -> torch.Tensor:
    if time_s.ndim == 1:
        time = time_s
    elif time_s.ndim == 2:
        if time_s.shape != (batch, 160):
            raise ValueError("batched time_s must have shape [B,160]")
        if not torch.equal(time_s, time_s[0].expand_as(time_s)):
            raise ValueError("batched time_s must contain one shared time vector")
        time = time_s[0]
    else:
        raise ValueError("time_s must have shape [160] or [B,160]")
    if time.numel() != 160:
        raise ValueError("AIS-MQFNO requires exactly 160 saved times")
    if not time.is_floating_point():
        raise ValueError("time_s must have a floating dtype")
    if not bool(torch.isfinite(time).all()):
        raise ValueError("time_s must contain only finite values")
    if not bool(torch.all(torch.diff(time) > 0)):
        raise ValueError("time_s must be strictly increasing")
    return time


def _validate_queries(query_xz: torch.Tensor, batch: int) -> None:
    if query_xz.ndim != 3 or query_xz.shape[0] != batch or query_xz.shape[-1] != 2:
        raise ValueError("query_xz must have shape [B,Q,2]")
    if not query_xz.is_floating_point() or not bool(torch.isfinite(query_xz).all()):
        raise ValueError("query_xz must contain finite floating values")
    if bool(torch.any((query_xz < 0) | (query_xz > 1))):
        raise ValueError("normalized query_xz coordinates must lie in [0,1]")


class DilatedTemporalResidual(nn.Module):
    """Depthwise dilation-1/2/4 correction for full spatiotemporal context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, dilation=1, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=4, groups=channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[3] != 160 or x.shape[-1] != self.channels:
            raise ValueError("expected x shaped [B,H,W,160,C]")
        batch, height, width, time_steps, channels = x.shape
        flat = x.reshape(batch * height * width, time_steps, channels).transpose(1, 2)
        corrected = self.net(flat).transpose(1, 2)
        return corrected.reshape(batch, height, width, time_steps, channels)


class TrajectoryHead(nn.Module):
    """Decode fused query features into all 160 physical-time samples."""

    def __init__(self, in_dim: int, temporal_dim: int, modes: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.input = nn.Linear(self.in_dim + 2, temporal_dim)
        self.operator = NonUniformTemporalOperator(temporal_dim, modes)
        self.output = nn.Linear(temporal_dim, 1)

    def forward(
        self,
        fused: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        if fused.ndim != 4 or fused.shape[2] != 160 or fused.shape[-1] != self.in_dim:
            raise ValueError("fused features must have shape [B,Q,160,in_dim]")
        time = _shared_full160_time(time_s, fused.shape[0])
        return self._forward_validated(fused, _make_validated_time_grid(time))

    def _forward_validated(
        self, fused: torch.Tensor, time_grid: _ValidatedTimeGrid
    ) -> torch.Tensor:
        if fused.ndim != 4 or fused.shape[2] != 160 or fused.shape[-1] != self.in_dim:
            raise ValueError("fused features must have shape [B,Q,160,in_dim]")
        time = _require_validated_time_grid(time_grid)
        if time.shape != (160,):
            raise ValueError("AIS-MQFNO requires exactly 160 saved times")
        tau = ((time - time[0]) / (time[-1] - time[0])).to(fused)
        time_features = torch.stack((tau, tau.square()), dim=-1)
        time_features = time_features[None, None].expand(
            fused.shape[0], fused.shape[1], -1, -1
        )
        hidden = self.input(torch.cat((fused, time_features), dim=-1))
        return self.output(self.operator._forward_validated(hidden, time_grid))


class AISMQFNO(nn.Module):
    """Global full160 propagation plus bounded native-grid query decoding."""

    supports_physical_spacing = True
    supports_validated_time_grid = True

    def __init__(
        self,
        global_in_features: int,
        native_in_channels: int,
        spatial_width: int = 24,
        spatial_modes: int = 24,
        temporal_modes: int = 32,
        local_dim: int = 16,
        fusion_dim: int = 32,
        halo_size: int = 17,
        spatial_layers: int = 3,
        activation_checkpointing: bool = False,
        spatial_chunk_size: int = 0,
        local_encoder_kind: str = "single",
        dispersion_head: str = "none",
        velocity_mean: float | None = None,
        velocity_std: float | None = None,
    ) -> None:
        super().__init__()
        global_in_features = _positive_integer(
            "global_in_features", global_in_features
        )
        native_in_channels = _positive_integer(
            "native_in_channels", native_in_channels
        )
        spatial_width = _positive_integer("spatial_width", spatial_width)
        spatial_modes = _positive_integer("spatial_modes", spatial_modes)
        temporal_modes = _positive_integer(
            "temporal_modes", temporal_modes, maximum=160
        )
        local_dim = _positive_integer("local_dim", local_dim)
        fusion_dim = _positive_integer("fusion_dim", fusion_dim)
        halo_size = _positive_integer("halo_size", halo_size)
        if halo_size % 2 == 0:
            raise ValueError("halo_size must be a positive odd integer")
        spatial_layers = _positive_integer("spatial_layers", spatial_layers)
        spatial_chunk_size = _nonnegative_integer(
            "spatial_chunk_size", spatial_chunk_size
        )
        if not isinstance(activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be a bool")
        if type(local_encoder_kind) is not str or local_encoder_kind not in (
            "single",
            "multiscale_9_25",
        ):
            raise ValueError(
                "local_encoder_kind must be 'single' or 'multiscale_9_25'"
            )
        if local_encoder_kind == "multiscale_9_25":
            if local_dim != 24:
                raise ValueError(
                    "local_dim must be 24 for local_encoder_kind='multiscale_9_25'"
                )
            if fusion_dim != 48:
                raise ValueError(
                    "fusion_dim must be 48 for local_encoder_kind='multiscale_9_25'"
                )
            if halo_size != 25:
                raise ValueError(
                    "halo_size must be 25 to record the maximum fixed multiscale halo"
                )
        if type(dispersion_head) is not str or dispersion_head not in (
            "none",
            "phase_residual_24",
        ):
            raise ValueError(
                "dispersion_head must be 'none' or 'phase_residual_24'"
            )
        if dispersion_head == "phase_residual_24" and (
            velocity_mean is None or velocity_std is None
        ):
            raise ValueError(
                "velocity_mean and velocity_std are required for dispersion_head"
            )
        if dispersion_head == "none" and (
            velocity_mean is not None or velocity_std is not None
        ):
            raise ValueError("velocity statistics are only valid for dispersion_head")

        self.global_in_features = global_in_features
        self.native_in_channels = native_in_channels
        self.spatial_width = spatial_width
        self.spatial_chunk_size = spatial_chunk_size
        self.activation_checkpointing = activation_checkpointing
        self.local_encoder_kind = local_encoder_kind
        self.halo_size = halo_size
        self.dispersion_head_kind = dispersion_head
        self.spatial_encoder = SpatialEncoder(
            self.global_in_features,
            self.spatial_width,
            spatial_modes,
            spatial_modes,
            spatial_layers=spatial_layers,
        )
        self.temporal_operator = NonUniformTemporalOperator(
            self.spatial_width, temporal_modes
        )
        self.dilated_temporal_residual = DilatedTemporalResidual(self.spatial_width)
        if self.local_encoder_kind == "single":
            self.local_encoder = LocalQueryEncoder(
                self.native_in_channels, local_dim, halo_size
            )
        else:
            self.local_encoder = MultiScaleLocalQueryEncoder(self.native_in_channels)
        self.fusion = nn.Sequential(
            nn.Linear(self.spatial_width + local_dim, fusion_dim), nn.GELU()
        )
        self.trajectory_head = TrajectoryHead(
            fusion_dim, fusion_dim, temporal_modes
        )
        self.dispersion_residual_head = (
            DispersionResidualHead(
                local_dim,
                hidden_dim=48,
                modes=24,
                velocity_mean=velocity_mean,
                velocity_std=velocity_std,
            )
            if dispersion_head == "phase_residual_24"
            else None
        )
        self.last_native_query_shape = (halo_size, halo_size)

    @staticmethod
    def sample_global_context(
        global_context: torch.Tensor, query_xz: torch.Tensor
    ) -> torch.Tensor:
        if global_context.ndim != 5 or global_context.shape[3] != 160:
            raise ValueError("global_context must have shape [B,H,W,160,C]")
        batch, height, width, time_steps, channels = global_context.shape
        _validate_queries(query_xz, batch)
        if global_context.device != query_xz.device:
            raise ValueError("global_context and query_xz must be on the same device")
        field = global_context.permute(0, 3, 4, 1, 2).reshape(
            batch * time_steps, channels, height, width
        )
        center = query_xz.to(dtype=global_context.dtype).mul(2).sub(1)
        # grid_sample coordinate order is horizontal z, then vertical x.
        grid = torch.stack((center[..., 1], center[..., 0]), dim=-1)
        grid = grid[:, None].expand(-1, time_steps, -1, -1)
        grid = grid.reshape(batch * time_steps, 1, query_xz.shape[1], 2)
        sampled = F.grid_sample(
            field, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled[:, :, 0].reshape(
            batch, time_steps, channels, query_xz.shape[1]
        ).permute(0, 3, 1, 2).contiguous()

    def encode_global(
        self,
        global_inputs: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        if global_inputs.ndim != 5 or global_inputs.shape[3] != 160:
            raise ValueError("global_inputs must have shape [B,Hg,Wg,160,Cg]")
        if global_inputs.shape[-1] != self.global_in_features:
            raise ValueError(
                f"global input features must equal {self.global_in_features}, got {global_inputs.shape[-1]}"
            )
        time = _shared_full160_time(time_s, global_inputs.shape[0])
        return self._encode_global_validated(
            global_inputs, _make_validated_time_grid(time)
        )

    def _encode_global_validated(
        self,
        global_inputs: torch.Tensor,
        time_grid: _ValidatedTimeGrid,
    ) -> torch.Tensor:
        time = _require_validated_time_grid(time_grid)
        if global_inputs.ndim != 5 or global_inputs.shape[3] != 160:
            raise ValueError("global_inputs must have shape [B,Hg,Wg,160,Cg]")
        if global_inputs.shape[-1] != self.global_in_features:
            raise ValueError(
                f"global input features must equal {self.global_in_features}, "
                f"got {global_inputs.shape[-1]}"
            )
        if time.shape != (160,):
            raise ValueError("validated time grid must contain exactly 160 saved times")
        spatial = self.spatial_encoder(
            global_inputs,
            spatial_chunk_size=self.spatial_chunk_size,
            use_checkpointing=self.activation_checkpointing,
        )
        batch, height, width, time_steps, channels = spatial.shape
        temporal = self.temporal_operator._forward_validated(
            spatial.reshape(batch, height * width, time_steps, channels),
            time_grid,
        ).reshape(batch, height, width, time_steps, channels)
        return temporal + self.dilated_temporal_residual(temporal)

    def decode_queries(
        self,
        global_context: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_s: torch.Tensor,
        *,
        dx_m: float | None = None,
        dz_m: float | None = None,
    ) -> torch.Tensor:
        if global_context.ndim != 5 or global_context.shape[-1] != self.spatial_width:
            raise ValueError("global_context has an invalid shape or channel count")
        time = _shared_full160_time(time_s, global_context.shape[0])
        return self._decode_queries_validated(
            global_context,
            native_static,
            query_xz,
            _make_validated_time_grid(time),
            dx_m=dx_m,
            dz_m=dz_m,
            spacing_host_validated=False,
        )

    def _decode_queries_validated(
        self,
        global_context: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_grid: _ValidatedTimeGrid,
        *,
        dx_m: float | None = None,
        dz_m: float | None = None,
        spacing_host_validated: bool = False,
    ) -> torch.Tensor:
        if global_context.ndim != 5 or global_context.shape[-1] != self.spatial_width:
            raise ValueError("global_context has an invalid shape or channel count")
        batch = global_context.shape[0]
        time = _require_validated_time_grid(time_grid)
        if time.shape != (160,):
            raise ValueError("AIS-MQFNO requires exactly 160 saved times")
        _validate_queries(query_xz, batch)
        if native_static.ndim != 4:
            raise ValueError("native_static must have shape [B,C,H,W]")
        if native_static.shape[0] != batch:
            raise ValueError("global and native batch sizes must match")
        if native_static.shape[1] != self.native_in_channels:
            raise ValueError(
                f"native_static channels must equal {self.native_in_channels}"
            )
        global_query = self.sample_global_context(global_context, query_xz)
        local_query = self.local_encoder(native_static, query_xz)
        local_over_time = local_query[:, :, None, :].expand(
            -1, -1, global_query.shape[2], -1
        )
        fused = self.fusion(torch.cat((global_query, local_over_time), dim=-1))
        trajectory = self.trajectory_head._forward_validated(
            fused, time_grid
        ).squeeze(-1)
        if self.dispersion_residual_head is None:
            return trajectory
        if dx_m is None or dz_m is None:
            raise ValueError("dispersion_head requires positive dx_m and dz_m spacing")
        center = query_xz.to(dtype=native_static.dtype).mul(2).sub(1)
        grid = torch.stack((center[..., 1], center[..., 0]), dim=-1)[:, None]
        velocity_hat = F.grid_sample(
            native_static[:, :1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, 0, 0]
        if isinstance(self.dispersion_residual_head, DispersionResidualHead):
            residual = self.dispersion_residual_head._forward_validated(
                local_query,
                velocity_hat,
                dx_m,
                dz_m,
                time_grid,
                spacing_host_validated=spacing_host_validated,
            )
        else:
            residual = self.dispersion_residual_head(
                local_query, velocity_hat, dx_m, dz_m, time
            )
        return trajectory + residual

    def forward(
        self,
        global_inputs: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_s: torch.Tensor,
        *,
        dx_m: float | None = None,
        dz_m: float | None = None,
    ) -> torch.Tensor:
        if native_static.ndim == 4 and native_static.shape[0] != global_inputs.shape[0]:
            raise ValueError("global and native batch sizes must match")
        time = _shared_full160_time(time_s, global_inputs.shape[0])
        time_grid = _make_validated_time_grid(time)
        context = self._encode_global_validated(global_inputs, time_grid)
        return self._decode_queries_validated(
            context,
            native_static,
            query_xz,
            time_grid,
            dx_m=dx_m,
            dz_m=dz_m,
        )


__all__ = [
    "AISMQFNO",
    "DilatedTemporalResidual",
    "LocalQueryEncoder",
    "MultiScaleLocalQueryEncoder",
    "TrajectoryHead",
]
