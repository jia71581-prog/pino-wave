"""Joint source-isolated query, dense-field, and receiver training step."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from time import perf_counter

import torch

from ..losses import consistency_loss, dual_head_losses, query_data_loss
from .audit import LossDominanceMonitor, require_gradients


def _to_device(batch, device):
    return {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for name, value in vars(batch).items()}


def compose_objective(loss_config, *, query, dense, trace, consistency,
                      gradient, spatial_fft, trace_fft):
    """Keep receiver supervision auxiliary to the two wavefield objectives."""
    data = loss_config.query * query + loss_config.dense * dense
    auxiliary = (loss_config.trace * trace
                 + loss_config.gradient * gradient
                 + loss_config.spatial_fft * spatial_fft
                 + loss_config.trace_fft * trace_fft
                 + loss_config.consistency * consistency)
    return data + auxiliary, data, auxiliary


class JointDualHeadTrainer:
    required_prefixes = ("medium_encoder", "source_encoder", "query_head", "dense_decoder")

    def __init__(self, model, normalizer, optimizer, loss_config, *, device="cpu",
                 gradient_clip=1.0, audit_every=1):
        self.model = model.to(device); self.normalizer = normalizer; self.optimizer = optimizer
        self.loss_config = loss_config; self.device = torch.device(device)
        self.gradient_clip = float(gradient_clip); self.audit_every = int(audit_every)
        self.step = 0; self.dominance = LossDominanceMonitor()

    def _receiver_coordinates(self, values):
        indices = values["receiver_zx_indices"]
        batch, receivers, _ = indices.shape; time = values["time_s"]
        nz, nx = values["source_map"].shape[-2:]
        x = indices[..., 1].float() * (self.model.domain_x_m / max(nx - 1, 1))
        z = indices[..., 0].float() * (self.model.domain_z_m / max(nz - 1, 1))
        nt = time.numel()
        coords = torch.empty(batch, receivers, nt, 3, device=self.device)
        coords[..., 0] = x[..., None]; coords[..., 1] = z[..., None]; coords[..., 2] = time[None, None]
        return coords.reshape(batch, receivers * nt, 3)

    def _forward_losses(self, batch):
        values = _to_device(batch, self.device)
        source, amplitude = values["source_parameters"], values["source_parameters"][:, 4]
        shared = dict(velocity_mps=values["velocity_mps"], source=source,
                      source_map=values["source_map"], normalizer=self.normalizer,
                      record_to_medium=values["record_to_medium"])
        dense_times = values["time_s"][values["dense_time_indices"]]
        query_prediction = self.model.query_normalized(coords=values["query_coords"], **shared)
        dense_prediction = torch.cat([
            self.model.dense_normalized(
                time_s=dense_times[:, start:start + self.model.dense_decoder.time_block], **shared
            )
            for start in range(0, dense_times.shape[1], self.model.dense_decoder.time_block)
        ], dim=1)
        receiver_shape = values["receiver_target"].shape
        receiver_prediction = self.model.query_normalized(
            coords=self._receiver_coordinates(values), chunk_size=4096, **shared
        ).reshape(receiver_shape)
        dense_target = self.normalizer.encode_pressure(values["dense_target"], amplitude[:, None, None, None])
        query_target = self.normalizer.encode_pressure(values["query_target"], amplitude[:, None])
        receiver_target = self.normalizer.encode_pressure(values["receiver_target"], amplitude[:, None, None])
        query_loss, query_rel = query_data_loss(query_prediction.float(), query_target, values["sample_probability"])
        structured = dual_head_losses(
            dense_prediction.float(), dense_target, receiver_prediction.float(), receiver_target,
            gradient_weight=self.loss_config.gradient,
            spatial_fft_weight=self.loss_config.spatial_fft,
            trace_fft_weight=self.loss_config.trace_fft,
        )
        # Compare both heads at the exact receiver/time cells represented by dense supervision.
        b = torch.arange(source.shape[0], device=self.device)[:, None, None]
        t = torch.arange(dense_prediction.shape[1], device=self.device)[None, :, None]
        z = values["receiver_zx_indices"][..., 0][:, None, :]
        x = values["receiver_zx_indices"][..., 1][:, None, :]
        dense_receiver = dense_prediction[b, t, z, x].transpose(1, 2)
        selected_trace = receiver_prediction.gather(2, values["dense_time_indices"][:, None].expand(-1, receiver_shape[1], -1))
        selected_target = receiver_target.gather(2, values["dense_time_indices"][:, None].expand(-1, receiver_shape[1], -1))
        agreement = consistency_loss(selected_trace, dense_receiver, selected_target)
        total, data, auxiliary = compose_objective(
            self.loss_config,
            query=query_loss, dense=structured.dense_data, trace=structured.trace_data,
            consistency=agreement, gradient=structured.gradient,
            spatial_fft=structured.spatial_spectrum, trace_fft=structured.trace_spectrum,
        )
        return total, data, auxiliary, {
            "loss_total": total.detach(), "loss_query": query_loss.detach(),
            "loss_dense": structured.dense_data.detach(), "loss_trace": structured.trace_data.detach(),
            "loss_gradient": structured.gradient.detach(), "loss_spatial_fft": structured.spatial_spectrum.detach(),
            "loss_trace_fft": structured.trace_spectrum.detach(), "loss_consistency": agreement.detach(),
            "query_relative_l2": query_rel.detach(), "dense_relative_l2": structured.dense_relative_l2.detach(),
            "trace_relative_l2": structured.trace_relative_l2.detach(),
        }

    def run_step(self, batch, *, train=True):
        started = perf_counter(); self.model.train(train)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        context = torch.autocast("cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
        with torch.set_grad_enabled(train), context:
            total, data, auxiliary, metrics = self._forward_losses(batch)
        gradient_norms = {}
        if train:
            total.backward()
            if self.step % self.audit_every == 0:
                gradient_norms = require_gradients(self.model, self.required_prefixes)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step(); self.step += 1
            metrics["auxiliary_data_ratio"] = self.dominance.update(float(data.detach()), float(auxiliary.detach()))
        elapsed = max(perf_counter() - started, 1e-9)
        result = {key: float(value) for key, value in metrics.items()}
        result.update({f"gradient_norm/{key}": value for key, value in gradient_norms.items()})
        result["records_per_second"] = len(batch.source_parameters) / elapsed
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        return result

    def run_lbfgs_step(self, batch_or_batches):
        """Perform a deterministic full-batch FP32 closure via microbatch accumulation."""
        if not isinstance(self.optimizer, torch.optim.LBFGS):
            raise TypeError("run_lbfgs_step requires torch.optim.LBFGS")
        started = perf_counter(); self.model.train(True)
        closure_evaluations = 0; gradient_norms = {}
        batches = (tuple(batch_or_batches) if isinstance(batch_or_batches, (list, tuple))
                   else (batch_or_batches,))
        record_count = sum(len(current.source_parameters) for current in batches)

        def closure():
            nonlocal closure_evaluations, gradient_norms
            self.optimizer.zero_grad(set_to_none=True)
            # Strong-Wolfe comparisons need stable FP32 objective values.
            objective = torch.zeros((), device=self.device)
            for current in batches:
                with torch.enable_grad():
                    total, _, _, _ = self._forward_losses(current)
                weight = len(current.source_parameters) / record_count
                (total * weight).backward()
                objective = objective + total.detach() * weight
            if closure_evaluations == 0 and self.step % self.audit_every == 0:
                gradient_norms = require_gradients(self.model, self.required_prefixes)
            closure_evaluations += 1
            return objective

        self.optimizer.step(closure); self.step += 1
        metrics = {f"gradient_norm/{key}": value for key, value in gradient_norms.items()}
        metrics["closure_evaluations"] = closure_evaluations
        metrics["records_per_second"] = record_count / max(perf_counter() - started, 1e-9)
        metrics["peak_memory_bytes"] = (torch.cuda.max_memory_allocated(self.device)
                                        if self.device.type == "cuda" else 0)
        return metrics
