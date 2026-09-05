# Aligned-time train-only ablation status

Validation and test_id truth remain sealed.

| Run | Status | Attempt progress | Baseline | Best candidate | Candidate improvement |
|---|---:|---:|---:|---:|---:|
| r5e_aligned_control | rejected | 8:140/140 | 0.277177996021 | 0.277183779621 | -5.78359958731e-06 |
| r5f_uniform_replay | pending | 5:92/140 | 0.277177996021 | 0.277208883044 | -3.08870227413e-05 |
| r5g_dropout005 | not_started | - | - | - | - |

## Paired candidate differences

- r5f_uniform_replay: LRx1=+4.825e-05, LRx0.5=-1.012e-05, LRx0.25=+2.410e-05, LRx0.125=+1.605e-05
- r5g_dropout005: pending
