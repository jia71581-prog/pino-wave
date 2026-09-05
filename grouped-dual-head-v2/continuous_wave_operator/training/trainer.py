from __future__ import annotations

import json
import math
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.nn import functional as F

from ..config import TrainingConfig
from ..data.dataset import ContinuousQueryBatch
from ..data.query_sampling import QuerySampler
from ..model import ContinuousWaveOperator
from .adaptive_sampling import HierarchicalAIS
from .checkpoint import load_checkpoint, save_checkpoint
from .losses import finite_difference_wave_residual, query_pressure_loss, spectral_trace_loss
from .validation import validation_metrics


class QueryDataset(Protocol):
    def __len__(self) -> int: ...
    def sample_queries(
        self,
        local_indices: list[int] | np.ndarray,
        sampler: QuerySampler,
        sampling_probabilities: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> ContinuousQueryBatch: ...


class Trainer:
    def __init__(
        self,
        *,
        model: ContinuousWaveOperator,
        train_dataset: QueryDataset,
        validation_dataset: QueryDataset,
        config: TrainingConfig,
        output_dir: str | Path,
        device: torch.device,
        config_digest: str,
        dataset_digest: str,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.device = device
        self.config_digest = config_digest
        self.dataset_digest = dataset_digest
        if config.data_workers > 0:
            enable_parallel = getattr(train_dataset, "enable_parallel_loading", None)
            if enable_parallel is None:
                raise ValueError("data_workers requires a parallel-capable training dataset")
            enable_parallel(config.data_workers)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_steps
        )
        medium_types = getattr(train_dataset, "medium_types", None)
        family_ids = None
        if medium_types is not None:
            if len(medium_types) != len(train_dataset):
                raise ValueError("dataset medium_types must match its sample count")
            family_names = sorted(set(medium_types))
            family_lookup = {name: index for index, name in enumerate(family_names)}
            family_ids = torch.tensor([family_lookup[name] for name in medium_types], dtype=torch.long)
        self.ais = HierarchicalAIS(
            sample_count=len(train_dataset),
            time_bins=config.time_bins,
            spatial_shape=config.spatial_ais_shape,
            seed=config.seed,
            family_ids=family_ids,
        )
        self.global_step = 0
        self.epoch = 0
        self.best_relative_l2 = math.inf
        self._request_sequence = 0
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=config.prefetch_batches,
            thread_name_prefix="hdf5-prefetch",
        )
        self._prefetch_futures: deque[
            Future[tuple[torch.Tensor, ContinuousQueryBatch]]
        ] = deque()

    def _move(self, batch: ContinuousQueryBatch) -> ContinuousQueryBatch:
        return ContinuousQueryBatch(
            velocity_mps=batch.velocity_mps.to(self.device),
            source_map=batch.source_map.to(self.device),
            source_parameters=batch.source_parameters.to(self.device),
            query_coords=batch.query_coords.to(self.device),
            target_pressure=batch.target_pressure.to(self.device),
            sample_indices=batch.sample_indices,
            sample_ids=batch.sample_ids,
            group_ids=batch.group_ids,
            medium_types=batch.medium_types,
        )

    def _query_cells(
        self, query_coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time = query_coords[:, 0, :, 2].detach().cpu()
        z = query_coords[:, 0, :, 1].detach().cpu()
        x = query_coords[:, 0, :, 0].detach().cpu()
        time_bins = torch.floor(time / self.model.domain.t_end_s * self.config.time_bins).long()
        time_bins.clamp_(0, self.config.time_bins - 1)
        height, width = self.config.spatial_ais_shape
        z_cells = torch.floor(z / self.model.domain.lz_m * height).long().clamp_(0, height - 1)
        x_cells = torch.floor(x / self.model.domain.lx_m * width).long().clamp_(0, width - 1)
        return time_bins, torch.stack((z_cells, x_cells), dim=-1)

    def _prepare_training_request(
        self,
    ) -> tuple[torch.Tensor, list[tuple[np.ndarray, np.ndarray]], QuerySampler]:
        local_indices = self.ais.sample_indices(self.config.batch_size, epoch=self.epoch)
        probabilities = self.ais.probabilities(epoch=self.epoch)
        frame_count = int(getattr(self.train_dataset, "time_s", np.empty(401)).size)
        frame_bins = np.floor(np.arange(frame_count) / frame_count * self.config.time_bins).astype(np.int64)
        sampling_probabilities = [
            (
                probabilities.time[index, frame_bins].numpy(),
                probabilities.spatial[index].numpy(),
            )
            for index in local_indices.tolist()
        ]
        sampler = QuerySampler(
            seed=self.config.seed + self._request_sequence,
            time_frames=self.config.time_frames,
            points_per_frame=self.config.points_per_frame,
        )
        self._request_sequence += 1
        return local_indices, sampling_probabilities, sampler

    def _load_training_request(
        self,
        request: tuple[
            torch.Tensor,
            list[tuple[np.ndarray, np.ndarray]],
            QuerySampler,
        ],
    ) -> tuple[torch.Tensor, ContinuousQueryBatch]:
        local_indices, sampling_probabilities, sampler = request
        batch = self.train_dataset.sample_queries(
            local_indices.tolist(), sampler, sampling_probabilities=sampling_probabilities
        )
        return local_indices, batch

    def _fill_prefetch_queue(self) -> None:
        while len(self._prefetch_futures) < self.config.prefetch_batches:
            self._prefetch_futures.append(
                self._prefetch_executor.submit(
                    self._load_training_request,
                    self._prepare_training_request(),
                )
            )

    def _next_training_batch(self) -> tuple[torch.Tensor, ContinuousQueryBatch]:
        self._fill_prefetch_queue()
        local_indices, batch = self._prefetch_futures.popleft().result()
        self._fill_prefetch_queue()
        return local_indices, batch

    def train_step(self) -> dict[str, float]:
        local_indices, loaded_batch = self._next_training_batch()
        batch = self._move(loaded_batch)
        probabilities = self.ais.probabilities(epoch=self.epoch)
        time_bins, spatial_cells = self._query_cells(batch.query_coords)
        sample_probability = probabilities.sample[local_indices][:, None]
        time_probability = probabilities.time[local_indices[:, None], time_bins]
        spatial_probability = probabilities.spatial[
            local_indices[:, None], spatial_cells[..., 0], spatial_cells[..., 1]
        ]
        joint_probability = (sample_probability * time_probability * spatial_probability)[:, None].to(
            self.device
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        medium = self.model.encode_medium(batch.velocity_mps)
        sources = self.model.encode_sources(medium, batch.source_map, batch.source_parameters)
        prediction = self.model.query(medium, sources, batch.query_coords)
        data_loss = query_pressure_loss(
            prediction, batch.target_pressure, probabilities=joint_probability
        )
        physics_fraction = min(
            max((self.global_step + 1 - self.config.stage1_steps) / self.config.physics_ramp_steps, 0.0),
            1.0,
        )
        spectral_loss = data_loss.new_zeros(())
        pde_loss = data_loss.new_zeros(())
        if physics_fraction > 0.0 and self.config.spectral_weight > 0.0:
            batch_size, shots, _ = prediction.shape
            time_frames, receiver_count = self.config.time_frames, self.config.points_per_frame
            predicted_traces = prediction.reshape(batch_size, shots, time_frames, receiver_count).permute(0, 1, 3, 2)
            target_traces = batch.target_pressure.reshape(
                batch_size, shots, time_frames, receiver_count
            ).permute(0, 1, 3, 2)
            spectral_loss = spectral_trace_loss(predicted_traces, target_traces)
        if physics_fraction > 0.0 and self.config.pde_weight > 0.0:
            count = min(self.config.pde_queries, batch.query_coords.shape[2])
            collocation = batch.query_coords[:, :, :count].clone()
            space_step = self.config.pde_space_step_m
            time_step = self.config.pde_time_step_s
            collocation[..., 0].clamp_(space_step, self.model.domain.lx_m - space_step)
            collocation[..., 1].clamp_(space_step, self.model.domain.lz_m - space_step)
            collocation[..., 2].clamp_(time_step, self.model.domain.t_end_s - time_step)
            grid = 2.0 * collocation[..., :2] / collocation.new_tensor(
                [self.model.domain.lx_m, self.model.domain.lz_m]
            ) - 1.0
            sampled_velocity = F.grid_sample(
                batch.velocity_mps,
                grid.reshape(grid.shape[0], -1, 1, 2),
                align_corners=True,
                mode="bilinear",
                padding_mode="border",
            ).squeeze(-1).reshape(grid.shape[0], 1, count)
            residual_pde = finite_difference_wave_residual(
                lambda coordinates: self.model.query(medium, sources, coordinates),
                collocation,
                sampled_velocity,
                dx_m=space_step,
                dz_m=space_step,
                dt_s=time_step,
            )
            source_xy = batch.source_parameters[..., :2]
            distance = torch.linalg.vector_norm(collocation[..., :2] - source_xy[:, :, None], dim=-1)
            mask = distance >= 30.0
            if bool(mask.any()):
                target_scale = batch.target_pressure.square().mean().sqrt().detach().clamp_min(1.0e-12)
                pde_loss = (residual_pde[mask] * time_step * time_step / target_scale).square().mean()
        loss = (
            data_loss
            + physics_fraction * self.config.spectral_weight * spectral_loss
            + physics_fraction * self.config.pde_weight * pde_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.scheduler.step()
        residual = (prediction.detach() - batch.target_pressure).abs()[:, 0].cpu()
        keep = min(residual.shape[1], 256)
        self.ais.update(
            local_indices,
            time_bins[:, :keep],
            spatial_cells[:, :keep],
            residual[:, :keep],
        )
        self.global_step += 1
        self.epoch = self.global_step * self.config.batch_size // max(len(self.train_dataset), 1)
        return {
            "step": float(self.global_step),
            "epoch": float(self.epoch),
            "loss": float(loss.detach()),
            "data_loss": float(data_loss.detach()),
            "spectral_loss": float(spectral_loss.detach()),
            "pde_loss": float(pde_loss.detach()),
            "physics_fraction": float(physics_fraction),
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        count = min(self.config.validation_samples, len(self.validation_dataset))
        sampler = QuerySampler(
            seed=self.config.seed + 100_000,
            time_frames=self.config.time_frames,
            points_per_frame=self.config.points_per_frame,
        )
        batch = self._move(self.validation_dataset.sample_queries(list(range(count)), sampler))
        self.model.eval()
        prediction = self.model(
            batch.velocity_mps,
            batch.source_map,
            batch.source_parameters,
            batch.query_coords,
            chunk_size=4096,
        )
        return validation_metrics(
            prediction[:, 0].cpu(), batch.target_pressure[:, 0].cpu(), medium_types=batch.medium_types
        )

    def _save(self, name: str) -> None:
        save_checkpoint(
            self.output_dir / "checkpoints" / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ais=self.ais,
            normalization={"pressure_scale": "per_batch_rms"},
            config_digest=self.config_digest,
            dataset_digest=self.dataset_digest,
            epoch=self.epoch,
            global_step=self.global_step,
        )

    def resume(self, path: str | Path) -> None:
        restored = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_config_digest=self.config_digest,
            expected_dataset_digest=self.dataset_digest,
            map_location=self.device,
        )
        self.epoch = restored.epoch
        self.global_step = restored.global_step
        self.ais = restored.ais
        self._request_sequence = self.global_step

    def fit(self) -> dict[str, float]:
        last: dict[str, float] = {}
        metrics_path = self.output_dir / "metrics.jsonl"
        while self.global_step < self.config.max_steps:
            last = self.train_step()
            print(json.dumps(last, sort_keys=True), flush=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(last, sort_keys=True) + "\n")
            if self.global_step % self.config.checkpoint_every == 0:
                self._save("last.pt")
            if self.global_step % self.config.validate_every == 0:
                validation = self.validate()
                event = {"step": self.global_step, "validation": validation}
                print(json.dumps(event, sort_keys=True), flush=True)
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                if validation["relative_l2"] < self.best_relative_l2:
                    self.best_relative_l2 = validation["relative_l2"]
                    self._save("best.pt")
        self._save("last.pt")
        self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
        return last
