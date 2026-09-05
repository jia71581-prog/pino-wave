from dataclasses import dataclass


@dataclass
class StageSchedule:
    stage1_steps: int = 1000
    stage2_steps: int = 1000
    pde_ramp_steps: int = 1000

    def at(self, step: int) -> str:
        if step < self.stage1_steps:
            return "data_pretrain"
        if step < self.stage1_steps + self.stage2_steps:
            return "dual_head_alignment"
        return "physics_finetune"

    def pde_weight(self, step: int, maximum: float) -> float:
        start = self.stage1_steps + self.stage2_steps
        if step <= start:
            return 0.0
        return float(maximum) * min(1.0, (step - start) / max(1, self.pde_ramp_steps))
