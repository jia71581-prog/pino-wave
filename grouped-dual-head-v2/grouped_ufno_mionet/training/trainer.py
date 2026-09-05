from __future__ import annotations

from dataclasses import dataclass
import torch

from .losses import grouped_operator_loss


@dataclass
class TrainerState:
    global_step: int = 0
    optimizer_steps: int = 0


class GroupedTrainer:
    """Four-block gradient accumulation trainer.

    Query blocks are consumed sequentially, so a macro-batch reads each frame
    once and updates the optimizer exactly once after all blocks.
    """
    def __init__(self, model, config=None, *, optimizer=None, device=None):
        self.model = model
        self.device = torch.device(device or next(model.parameters()).device)
        self.model.to(self.device)
        lr = float(getattr(getattr(config, "train", config), "learning_rate", 2e-4))
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr)
        self.state = TrainerState()
        self.amp_enabled = self.device.type == "cuda"
        if self.amp_enabled:
            # RTX 3090 tensor cores benefit from FP16 AMP and TF32 matmuls;
            # both preserve FP32 master weights in the optimizer.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    @staticmethod
    def _blocks(batch):
        blocks = getattr(batch, "query_blocks", None)
        if blocks:
            return blocks
        if isinstance(batch, dict) and "query_blocks" in batch:
            return batch["query_blocks"]
        raise ValueError("macro batch must provide query_blocks")

    def train_macro_batch(self, batch):
        self.model.train()
        def to_device(value, dtype):
            value = torch.as_tensor(value, dtype=dtype)
            return value.to(self.device, non_blocking=value.is_pinned())

        velocity = to_device(batch.velocity_mps, torch.float32)
        source = to_device(batch.source_parameters, torch.float32)
        mapping = to_device(batch.record_to_medium, torch.long)
        blocks = self._blocks(batch)
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        # All blocks use one graph encoding; the model API currently exposes
        # the cache through prepare/query_encoded when available.
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
            cache = self.model.prepare(velocity, source, record_to_medium=mapping) if hasattr(self.model, "prepare") else None
            for block_index, block in enumerate(blocks):
                coords = to_device(block.coords, torch.float32)
                target = to_device(block.targets, torch.float32)
                if coords.ndim >= 4:
                    coords = coords.reshape(coords.shape[0], -1, 3)
                if target.ndim >= 2:
                    target = target.reshape(target.shape[0], -1)
                pred = (self.model.query_encoded(cache, coords) if cache is not None
                        else self.model.query_pressure(velocity, source, coords, record_to_medium=mapping, chunk_size=4096))
                loss, _ = grouped_operator_loss(pred, target, group_ids=getattr(batch, "sample_id", None))
                self.scaler.scale(loss / len(blocks)).backward(retain_graph=block_index < len(blocks) - 1)
                total = total + loss.detach()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.state.global_step += 1
        self.state.optimizer_steps += 1
        return {"loss": float(total / len(blocks)), "optimizer_steps": self.state.optimizer_steps}
