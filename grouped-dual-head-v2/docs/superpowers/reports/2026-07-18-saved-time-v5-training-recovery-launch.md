# Saved-Time V5 Training-Recovery Launch

Date: 2026-07-18

## Verified implementation gates

- The full V4 regression suite passes with 70 tests before launch.
- Every production epoch covers all 2,240 training records in 188 macros and 47 effective-batch-48 updates.
- Exact-time coverage is minimum 120 indices per record at epoch 30 and median 200 at epoch 50.
- The planned pre-onset fraction is 0.0496454 and interpolated requests are zero.
- Microbatch 16 exceeds 24 GB. Dense/source/fusion passes with physical microbatch 12 at about 20.2 GB; coordinate/travel unfreezing requires microbatch 8, and the medium-backbone stage is registered at microbatch 4. Effective batch remains 48 throughout.
- The CUDA smoke produces finite nonzero gradients for the dense decoder, source encoder, and fusion groups.

## Active gated pipeline

- Pilot PID at launch: `2593954`
- Supervisor PID at launch: `2596752`
- Pilot log: `/data/jiayh/saved_time_v5_training_recovery_batch48/pilot_launcher.log`
- Pilot checkpoints: `/data/jiayh/saved_time_v5_training_recovery_batch48/pilot/checkpoints/`
- Supervisor state: `/data/jiayh/saved_time_v5_training_recovery_batch48/supervisor_status.json`
- Production log after a passing pilot: `/data/jiayh/saved_time_v5_training_recovery_batch48/launcher.log`
- Final 480×401 evaluation log: `/data/jiayh/saved_time_v5_training_recovery_batch48/evaluation_launcher.log`

Five-second live sampling after launch showed 100% SM utilization, 342–348 W power, and approximately 20.0 GiB framebuffer use. No accuracy claim is made until the fixed-panel pilot, 50-epoch production run, and sealed all-time evaluation complete.

The prior dense-only ASAM run completed normally. Its best fixed-panel relative L2 was 0.5577154 and its preregistered improvement gate failed; it was not interrupted by this launch.
