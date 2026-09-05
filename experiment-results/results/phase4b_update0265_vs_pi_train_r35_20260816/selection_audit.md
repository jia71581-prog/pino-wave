# Checkpoint selection audit

The user-selected approximately 4% checkpoint is the archived Phase4b update 265 weight.

- Checkpoint: `results/helmholtz_g3_phase4b_vrba_frames32_pf420_ep5_20260807T031404Z/checkpoints/update_0265.pt`
- SHA-256: `f6efbb81dd1e9baab0eb34b32b125e2cb58cd3b3292ee0db33a29194c84bfea1`
- Archived manifest: `a20c9a65abbc65294062af443e2ceae241ead66450f940f652e2d95aaa0aa92b`
- Repaired evaluation manifest: `55fbffa9a66b0cb547657d2d5cd8cc140c4f7d970e37f3828144e7778d182e09`
- Strict transfer: all 320 tensors matched; no tensor was initialized or omitted.
- Historical G3 selection metric: `0.040691912995691505` on the fixed 32-frame triplet.
- Historical all-401 G3 metric: `0.04165225127825517`.
- Same-six-record mean complete-future metric: `0.10599196605053145`.
- Same-six-record PI-DeepONet metric: `0.6156354159371821`.
- External numerical background required: yes.
- Instance adaptation applied in this result: no. The separately reported DFO+RCFA row retains the paper's instance-fine-tuning evidence without relabeling the Phase4b parent.

